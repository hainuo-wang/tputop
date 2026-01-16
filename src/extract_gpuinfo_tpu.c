/*
 *
 * Copyright (C) 2025 Robert Dyro <robert.dyro@gmail.com>
 *
 * This file is part of Nvtop
 *
 * Nvtop is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Nvtop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with nvtop.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "nvtop/extract_gpuinfo_common.h"
#include "nvtop/time.h"

#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <pwd.h>
#include <inttypes.h>
#include <pthread.h>
#include <signal.h>

/* Global exit flag for quick shutdown */
static volatile sig_atomic_t tpu_exit_requested = 0;

/* Environment variable to enable remote TPU pod monitoring */
#define NVTOP_TPU_POD_ENV "NVTOP_TPU_POD_FILE"
#define REMOTE_CACHE_VALIDITY_MS 50
#define MAX_REMOTE_HOSTS 64
#define MAX_HOST_LEN 64
#define DEFAULT_PODIPS_FILE "~/podips.txt"

/* TPU monitoring mode */
typedef enum {
  TPU_MODE_DEFAULT,  /* Local + remote (GCP metadata -> NVTOP_TPU_POD_FILE) */
  TPU_MODE_LOCAL,    /* Local TPU only */
  TPU_MODE_PODIPS    /* Only from podips file, no local */
} tpu_monitor_mode_t;

static tpu_monitor_mode_t tpu_monitor_mode = TPU_MODE_DEFAULT;
static char tpu_podips_path[512] = "";

struct gpu_info_tpu {
  struct gpu_info base;
  int device_id;
  bool is_remote;
  int remote_host_idx;
  int remote_device_id;
};

struct tpu_chip_usage_data {
  char name[8];
  int64_t device_id;
  int64_t memory_usage;
  int64_t total_memory;
  double duty_cycle_pct;
  int64_t pid;
  char cmdline[256];
  char user_name[64];
  unsigned cpu_percent;
  uint64_t host_mem;
};

struct remote_host {
  char ip[MAX_HOST_LEN];
  char tpu_model[32];  // TPU model name from remote host
  int tpu_count;
  struct tpu_chip_usage_data *usage_data;
  nvtop_time last_refresh;
  pthread_t refresh_thread;
  bool thread_running;
};

static struct remote_host *remote_hosts = NULL;
static int remote_host_count = 0;
static int total_remote_tpus = 0;
static bool remote_monitoring_enabled = false;

static bool gpuinfo_tpu_init(void);
static void gpuinfo_tpu_shutdown(void);
static const char *gpuinfo_tpu_last_error_string(void);
static bool gpuinfo_tpu_get_device_handles(struct list_head *devices, unsigned *count);
static void gpuinfo_tpu_populate_static_info(struct gpu_info *_gpu_info);
static void gpuinfo_tpu_refresh_dynamic_info(struct gpu_info *_gpu_info);
static void gpuinfo_tpu_get_running_processes(struct gpu_info *_gpu_info);
static bool is_cache_valid(void);
static bool refresh_tpu_cache(void);
static void reset_tpu_cache(bool);
static void free_ptr(void **ptr);
static bool load_remote_hosts(void);
static bool is_remote_cache_valid(int host_idx);
static bool refresh_remote_tpu_cache(int host_idx);
static void *refresh_remote_thread(void *arg);
static void refresh_all_remote_hosts_parallel(void);

/* Set exit flag for quick shutdown */
void gpuinfo_tpu_request_exit(void) {
  tpu_exit_requested = 1;
}

/* Set TPU monitoring mode */
void gpuinfo_tpu_set_mode(int mode, const char *podips_path) {
  tpu_monitor_mode = (tpu_monitor_mode_t)mode;
  if (podips_path && podips_path[0] != '\0') {
    strncpy(tpu_podips_path, podips_path, sizeof(tpu_podips_path) - 1);
    tpu_podips_path[sizeof(tpu_podips_path) - 1] = '\0';
  }
}

struct gpu_vendor gpu_vendor_tpu = {
    .init = gpuinfo_tpu_init,
    .shutdown = gpuinfo_tpu_shutdown,
    .last_error_string = gpuinfo_tpu_last_error_string,
    .get_device_handles = gpuinfo_tpu_get_device_handles,
    .populate_static_info = gpuinfo_tpu_populate_static_info,
    .refresh_dynamic_info = gpuinfo_tpu_refresh_dynamic_info,
    .refresh_running_processes = gpuinfo_tpu_get_running_processes,
    .name = "TPU",
};

__attribute__((constructor)) static void init_extract_gpuinfo_tpu(void) { 
  register_gpu_vendor(&gpu_vendor_tpu);
}

int64_t tpu_chip_count = -1;
static struct gpu_info_tpu *gpu_infos;

#define STRINGIFY(x) STRINGIFY_HELPER_(x)
#define STRINGIFY_HELPER_(x) #x

#define VENDOR_TPU 0x1ae0
#define VENDOR_TPU_STR STRINGIFY(VENDOR_TPU)

#define MAX(x, y) ((x >= y) ? (x) : (y))
#define MIN(x, y) ((x <= y) ? (x) : (y))

#define int64 long long

int (*_tpu_chip_count)(void);
int (*_tpu_metrics)(int port, int64 *device_ids, int64 *memory_usage, 
                    int64 *total_memory, double *duty_cycle_pct, int n);
int (*_tpu_pids)(int64 *pids, int n);

char *libname = "libtpuinfo.so";
// -1 means allowing libtpuinfo to select the default port
// env LIBTPUINFO_GRPC_PORT={int} allows setting the port via an environment variable
// $ env LIBTPUINFO_GRPC_PORT=8431 nvtop
int tpu_runtime_monitoring_port = -1;  

/* TPU info cache ------------------------------------------------------------------------------- */
struct tpu_chip_usage_data *latest_chips_usage_data = NULL;
nvtop_time last_cache_refresh;
int64 *_pids, *_device_ids, *_memory_usage, *_total_memory;
double* _duty_cycle_pct;

bool is_cache_valid(void) {
  nvtop_time current_time;
  nvtop_get_current_time(&current_time);
  uint64_t t_diff_ns = nvtop_difftime_u64(last_cache_refresh, current_time);
  return t_diff_ns < 50 * 1000 * 1000; // 50ms for 20fps
}

bool refresh_tpu_cache(void) {
  if (is_cache_valid()) return true;
  nvtop_get_current_time(&last_cache_refresh);
  if (tpu_chip_count <= 0) return false;
  if (_tpu_pids(_pids, tpu_chip_count) != 0) {
    reset_tpu_cache(false);
    return false;
  }
  for (int64_t i = 0; i < tpu_chip_count; i++) latest_chips_usage_data[i].pid = _pids[i];

  if (_tpu_metrics(tpu_runtime_monitoring_port, _device_ids, _memory_usage, _total_memory,
                   _duty_cycle_pct, tpu_chip_count) != 0) return false;
  for (int64_t i = 0; i < tpu_chip_count; i++) {
    latest_chips_usage_data[i].device_id = _device_ids[i];
    latest_chips_usage_data[i].memory_usage = _memory_usage[i];
    latest_chips_usage_data[i].total_memory = _total_memory[i];
    latest_chips_usage_data[i].duty_cycle_pct = _duty_cycle_pct[i];
  }
  return true;
}

void reset_tpu_cache(bool fully) {
  for (int64_t i = 0; i < tpu_chip_count; i++) {
    latest_chips_usage_data[i].memory_usage = 0;
    latest_chips_usage_data[i].duty_cycle_pct = 0;
    latest_chips_usage_data[i].pid = -1;
    if (fully) {
      snprintf(latest_chips_usage_data[i].name, sizeof(latest_chips_usage_data[i].name), "%s", "N/A");
      latest_chips_usage_data[i].device_id = 0;
      latest_chips_usage_data[i].total_memory = 0;
    }
  }
}
/* TPU info cache ------------------------------------------------------------------------------- */

/* TPU model detection -------------------------------------------------------------------------- */
/* Cache for local TPU model name (from GCP metadata) */
static char local_tpu_model[32] = "";
static bool local_tpu_model_initialized = false;

/* Parse TPU version from string (e.g., "v4-64" -> "TPU v4", "v6e-8" -> "TPU v6e") */
static bool parse_tpu_version(const char *str, char *out, size_t out_size) {
  const char *p = str;
  while (*p) {
    if (*p == 'v' && p[1] >= '0' && p[1] <= '9') {
      // Found vX pattern, extract version number
      int version = 0;
      const char *num_start = p + 1;
      while (*num_start >= '0' && *num_start <= '9') {
        version = version * 10 + (*num_start - '0');
        num_start++;
      }
      // Check for suffix (e, p, lite)
      const char *suffix = "";
      if (strncmp(num_start, "lite", 4) == 0 || *num_start == 'e') {
        suffix = "e";
      } else if (*num_start == 'p') {
        suffix = "p";
      }
      snprintf(out, out_size, "TPU v%d%s", version, suffix);
      return true;
    }
    p++;
  }
  return false;
}

/* Get TPU model from GCP metadata */
static const char* get_tpu_model_from_metadata(void) {
  if (local_tpu_model_initialized) {
    return local_tpu_model[0] ? local_tpu_model : "TPU";
  }
  local_tpu_model_initialized = true;

  char buf[128];

  // First try accelerator-type (e.g., "v4-64")
  FILE *fp = popen("curl -s -H 'Metadata-Flavor: Google' "
                   "'http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type' "
                   "2>/dev/null", "r");
  if (fp) {
    if (fgets(buf, sizeof(buf), fp)) {
      char *newline = strchr(buf, '\n');
      if (newline) *newline = '\0';
      parse_tpu_version(buf, local_tpu_model, sizeof(local_tpu_model));
    }
    pclose(fp);
  }

  // Fallback to machine-type if accelerator-type didn't work
  if (!local_tpu_model[0]) {
    fp = popen("curl -s -H 'Metadata-Flavor: Google' "
               "'http://metadata.google.internal/computeMetadata/v1/instance/machine-type' "
               "2>/dev/null", "r");
    if (fp) {
      if (fgets(buf, sizeof(buf), fp)) {
        char *newline = strchr(buf, '\n');
        if (newline) *newline = '\0';
        parse_tpu_version(buf, local_tpu_model, sizeof(local_tpu_model));
      }
      pclose(fp);
    }
  }

  return local_tpu_model[0] ? local_tpu_model : "TPU";
}

static const char* get_tpu_model_name(int device_id) {
  char path[128];
  snprintf(path, sizeof(path), "/sys/class/accel/accel%d/device/device", device_id);
  FILE *f = fopen(path, "r");
  if (!f) {
    // Fallback to GCP metadata
    return get_tpu_model_from_metadata();
  }

  char buf[32];
  if (!fgets(buf, sizeof(buf), f)) {
    fclose(f);
    return get_tpu_model_from_metadata();
  }
  fclose(f);

  unsigned int dev_id = 0;
  sscanf(buf, "%x", &dev_id);

  // Map device ID to TPU model name
  switch (dev_id) {
    case 0x0027: return "TPU v2";
    case 0x0056: return "TPU v3";
    case 0x005e: return "TPU v4";
    case 0x0063: return "TPU v5e";
    case 0x0062: return "TPU v5p";
    default: return get_tpu_model_from_metadata();
  }
}
/* TPU model detection -------------------------------------------------------------------------- */

/* Remote TPU support --------------------------------------------------------------------------- */
static char* get_home_directory(void) {
  char *home = getenv("HOME");
  if (home) return home;
  struct passwd *pw = getpwuid(getuid());
  if (pw) return pw->pw_dir;
  return NULL;
}

/* Get local IP address to filter out self from worker list */
static bool get_local_ip(char *local_ip, size_t len) {
  FILE *fp = popen("hostname -I 2>/dev/null | awk '{print $1}'", "r");
  if (!fp) return false;
  if (!fgets(local_ip, len, fp)) {
    pclose(fp);
    return false;
  }
  pclose(fp);
  char *nl = strchr(local_ip, '\n');
  if (nl) *nl = '\0';
  return strlen(local_ip) > 0;
}

/* Load worker IPs from GCP metadata service */
static bool load_from_metadata(void) {
  FILE *fp = popen("curl -s -H 'Metadata-Flavor: Google' "
                   "'http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-network-endpoints' "
                   "2>/dev/null", "r");
  if (!fp) return false;

  char output[2048];
  if (!fgets(output, sizeof(output), fp)) {
    pclose(fp);
    return false;
  }
  pclose(fp);

  // Remove trailing newline
  char *nl = strchr(output, '\n');
  if (nl) *nl = '\0';

  char local_ip[64] = {0};
  get_local_ip(local_ip, sizeof(local_ip));

  remote_hosts = (struct remote_host *)calloc(MAX_REMOTE_HOSTS, sizeof(struct remote_host));
  if (!remote_hosts) return false;

  // Parse: unknown:unknown:IP1,unknown:unknown:IP2,...
  char *saveptr = NULL;
  char *token = strtok_r(output, ",", &saveptr);
  remote_host_count = 0;

  while (token && remote_host_count < MAX_REMOTE_HOSTS) {
    // Extract IP from "unknown:unknown:IP"
    char *ip = strrchr(token, ':');
    if (ip) {
      ip++; // Skip the ':'
      // Skip local IP
      if (strlen(local_ip) == 0 || strcmp(ip, local_ip) != 0) {
        strncpy(remote_hosts[remote_host_count].ip, ip, MAX_HOST_LEN - 1);
        remote_hosts[remote_host_count].tpu_count = 0;
        remote_hosts[remote_host_count].usage_data = NULL;
        nvtop_get_current_time(&remote_hosts[remote_host_count].last_refresh);
        remote_hosts[remote_host_count].last_refresh =
            nvtop_substract_time(remote_hosts[remote_host_count].last_refresh, (nvtop_time){10, 0});
        remote_host_count++;
      }
    }
    token = strtok_r(NULL, ",", &saveptr);
  }

#ifndef NDEBUG
  fprintf(stderr, "TPU Pod: Found %d remote hosts (local: %s)\n", remote_host_count, local_ip);
#endif
  return remote_host_count > 0;
}

/* Load remote hosts from a file */
static bool load_from_file(const char *filepath) {
  char expanded_path[512];
  if (filepath[0] == '~') {
    char *home = get_home_directory();
    if (!home) return false;
    snprintf(expanded_path, sizeof(expanded_path), "%s%s", home, filepath + 1);
  } else {
    snprintf(expanded_path, sizeof(expanded_path), "%s", filepath);
  }

  FILE *f = fopen(expanded_path, "r");
  if (!f) return false;

  remote_hosts = (struct remote_host *)calloc(MAX_REMOTE_HOSTS, sizeof(struct remote_host));
  if (!remote_hosts) {
    fclose(f);
    return false;
  }

  char line[256];
  remote_host_count = 0;
  while (fgets(line, sizeof(line), f) && remote_host_count < MAX_REMOTE_HOSTS) {
    char *newline = strchr(line, '\n');
    if (newline) *newline = '\0';
    char *cr = strchr(line, '\r');
    if (cr) *cr = '\0';
    if (strlen(line) == 0) continue;

    strncpy(remote_hosts[remote_host_count].ip, line, MAX_HOST_LEN - 1);
    remote_hosts[remote_host_count].tpu_count = 0;
    remote_hosts[remote_host_count].usage_data = NULL;
    nvtop_get_current_time(&remote_hosts[remote_host_count].last_refresh);
    remote_hosts[remote_host_count].last_refresh =
        nvtop_substract_time(remote_hosts[remote_host_count].last_refresh, (nvtop_time){10, 0});
    remote_host_count++;
  }
  fclose(f);

  remote_monitoring_enabled = (remote_host_count > 0);
  return remote_monitoring_enabled;
}

static bool load_remote_hosts(void) {
  // TPU_MODE_LOCAL: no remote hosts
  if (tpu_monitor_mode == TPU_MODE_LOCAL) {
    return false;
  }

  // TPU_MODE_PODIPS: only load from podips file
  if (tpu_monitor_mode == TPU_MODE_PODIPS) {
    char path[1024];
    if (tpu_podips_path[0] == '\0') {
      // Default: ~/podips.txt
      snprintf(path, sizeof(path), "%s", DEFAULT_PODIPS_FILE);
    } else if (strchr(tpu_podips_path, '/') == NULL && tpu_podips_path[0] != '~') {
      // Just a name: convert to ~/name.txt
      snprintf(path, sizeof(path), "~/%s.txt", tpu_podips_path);
    } else {
      // Full path provided
      snprintf(path, sizeof(path), "%s", tpu_podips_path);
    }
    return load_from_file(path);
  }

  // TPU_MODE_DEFAULT: GCP metadata -> NVTOP_TPU_POD_FILE
  if (load_from_metadata()) {
    remote_monitoring_enabled = true;
    return true;
  }

  // Fallback: load from file specified by environment variable
  char *pod_file = getenv(NVTOP_TPU_POD_ENV);
  if (!pod_file) return false;

  return load_from_file(pod_file);
}

static bool is_remote_cache_valid(int host_idx) {
  if (host_idx < 0 || host_idx >= remote_host_count) return false;
  nvtop_time current_time;
  nvtop_get_current_time(&current_time);
  uint64_t t_diff_ns = nvtop_difftime_u64(remote_hosts[host_idx].last_refresh, current_time);
  return t_diff_ns < REMOTE_CACHE_VALIDITY_MS * 1000 * 1000;
}

static bool refresh_remote_tpu_cache(int host_idx) {
  if (host_idx < 0 || host_idx >= remote_host_count) return false;
  if (is_remote_cache_valid(host_idx)) return true;

  struct remote_host *host = &remote_hosts[host_idx];
  nvtop_get_current_time(&host->last_refresh);

  // SSH command - get TPU info with process details and model
  char cmd[4096];
  snprintf(cmd, sizeof(cmd),
           "ssh -o ConnectTimeout=2 -o BatchMode=yes -o StrictHostKeyChecking=no %s "
           "'MODEL=$(curl -s -m 1 -H \"Metadata-Flavor: Google\" "
           "\"http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type\" 2>/dev/null); "
           "python3 -c \""
           "import ctypes as C,os,sys;"
           "r=sys.argv[1] if len(sys.argv)>1 else \\\"\\\";"
           "import re;m=re.search(r\\\"v(\\d+)(lite|e|p)?\\\",r);s=m.group(2)or\\\"\\\";model=\\\"TPU v\\\"+m.group(1)+(\\\"e\\\" if s in(\\\"lite\\\",\\\"e\\\")else \\\"p\\\" if s==\\\"p\\\" else\\\"\\\")if m else\\\"TPU\\\";"
           "lib=C.CDLL(\\\"libtpuinfo.so\\\");"
           "lib.tpu_chip_count.restype=C.c_int;"
           "c=lib.tpu_chip_count();"
           "print(model+\\\"|\\\"+str(c)) if c<=0 else None;"
           "exit(0) if c<=0 else None;"
           "lib.tpu_pids.argtypes=[C.POINTER(C.c_longlong),C.c_int];"
           "lib.tpu_metrics.argtypes=[C.c_int,C.POINTER(C.c_longlong),C.POINTER(C.c_longlong),C.POINTER(C.c_longlong),C.POINTER(C.c_double),C.c_int];"
           "p=(C.c_longlong*c)();d=(C.c_longlong*c)();m=(C.c_longlong*c)();t=(C.c_longlong*c)();u=(C.c_double*c)();"
           "lib.tpu_pids(p,c);lib.tpu_metrics(-1,d,m,t,u,c);"
           "gi=lambda pid:(os.popen(f\\\"stat -c %%U /proc/{pid}\\\").read().strip() or \\\"N/A\\\","
           "os.popen(f\\\"ps -p {pid} -o pcpu=\\\").read().strip() or \\\"0\\\","
           "os.popen(f\\\"awk {{print\\\\\\$2*4096}} /proc/{pid}/statm\\\").read().strip() or \\\"0\\\","
           "os.popen(f\\\"cat /proc/{pid}/cmdline\\\").read().replace(chr(0),\\\" \\\")[:150].replace(\\\",\\\",\\\" \\\") or \\\"N/A\\\") if pid>0 else (\\\"N/A\\\",\\\"0\\\",\\\"0\\\",\\\"N/A\\\");"
           "R=[gi(p[i]) for i in range(c)];"
           "print(model+\\\"|\\\"+str(c)+\\\"|\\\"+\\\"|\\\".join([f\\\"{d[i]}~{m[i]}~{t[i]}~{u[i]:.2f}~{p[i]}~{R[i][0]}~{R[i][1]}~{R[i][2]}~{R[i][3]}\\\" for i in range(c)]))"
           "\" \"$MODEL\"' 2>/dev/null",
           host->ip);

  FILE *fp = popen(cmd, "r");
  if (!fp) return false;

  char output[4096];
  if (!fgets(output, sizeof(output), fp)) {
    pclose(fp);
    return false;
  }
  pclose(fp);

  // Parse output: model|count|dev_id~mem_used~mem_total~duty~pid~user~cpu~rss~cmd|...
  int count = 0;
  char *saveptr1 = NULL;

  // First token: TPU model
  char *token = strtok_r(output, "|", &saveptr1);
  if (!token) return false;
  strncpy(host->tpu_model, token, sizeof(host->tpu_model) - 1);
  host->tpu_model[sizeof(host->tpu_model) - 1] = '\0';

  // Second token: count
  token = strtok_r(NULL, "|", &saveptr1);
  if (!token) return false;
  count = atoi(token);
  if (count <= 0) return false;

  // Allocate/reallocate usage_data if needed
  if (host->tpu_count != count) {
    free(host->usage_data);
    host->usage_data = (struct tpu_chip_usage_data *)calloc(count, sizeof(struct tpu_chip_usage_data));
    if (!host->usage_data) return false;
    host->tpu_count = count;
  }

  for (int i = 0; i < count; i++) {
    token = strtok_r(NULL, "|", &saveptr1);
    if (!token) break;

    // Parse: dev_id~mem_used~mem_total~duty~pid~user~cpu~rss~cmdline (using ~ as delimiter)
    char *saveptr2 = NULL;
    char *field = strtok_r(token, "~", &saveptr2);
    if (field) host->usage_data[i].device_id = atoll(field);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) host->usage_data[i].memory_usage = atoll(field);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) host->usage_data[i].total_memory = atoll(field);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) host->usage_data[i].duty_cycle_pct = atof(field);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) host->usage_data[i].pid = atoll(field);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) strncpy(host->usage_data[i].user_name, field, sizeof(host->usage_data[i].user_name) - 1);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) host->usage_data[i].cpu_percent = (unsigned)atof(field);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) host->usage_data[i].host_mem = (uint64_t)atoll(field);
    field = strtok_r(NULL, "~", &saveptr2);
    if (field) strncpy(host->usage_data[i].cmdline, field, sizeof(host->usage_data[i].cmdline) - 1);
  }
  return true;
}

/* Thread function for parallel remote refresh */
static void *refresh_remote_thread(void *arg) {
  int host_idx = *(int *)arg;
  free(arg);
  if (!tpu_exit_requested) {
    refresh_remote_tpu_cache(host_idx);
  }
  if (host_idx >= 0 && host_idx < remote_host_count) {
    remote_hosts[host_idx].thread_running = false;
  }
  return NULL;
}

/* Refresh all remote hosts in parallel */
static void refresh_all_remote_hosts_parallel(void) {
  if (!remote_monitoring_enabled || tpu_exit_requested) return;

  for (int h = 0; h < remote_host_count; h++) {
    if (tpu_exit_requested) break;
    if (is_remote_cache_valid(h)) continue;
    if (remote_hosts[h].thread_running) continue;

    int *arg = malloc(sizeof(int));
    if (!arg) continue;
    *arg = h;
    remote_hosts[h].thread_running = true;
    if (pthread_create(&remote_hosts[h].refresh_thread, NULL,
                       refresh_remote_thread, arg) != 0) {
      remote_hosts[h].thread_running = false;
      free(arg);
    }
  }
}
/* Remote TPU support --------------------------------------------------------------------------- */

bool gpuinfo_tpu_init(void) {
  char* error_msg;
  nvtop_get_current_time(&last_cache_refresh);
  // invalidate cache by putting it in the past
  last_cache_refresh = nvtop_substract_time(last_cache_refresh, (nvtop_time){10, 0});

  // In PODIPS mode, skip local TPU detection entirely
  if (tpu_monitor_mode == TPU_MODE_PODIPS) {
    tpu_chip_count = 0;
    load_remote_hosts();
    if (!remote_monitoring_enabled) {
#ifndef NDEBUG
      fprintf(stderr, "PODIPS mode: no remote hosts configured.\n");
#endif
      return false;
    }
    return true;
  }

  // Load dynamic library symbols
  void *handle = dlopen(libname, RTLD_LAZY);
  if (!handle) {
      error_msg = dlerror();
#ifndef NDEBUG
      if (error_msg != NULL) fprintf(stderr, "TPU support error: %s\n", error_msg);
#endif
      return false;
  }

  // Resolve the necessary symbols within the library
  _tpu_chip_count = dlsym(handle, "tpu_chip_count");
  error_msg = dlerror();
  if (error_msg != NULL) {
#ifndef NDEBUG
      fprintf(stderr, "libtpuinfo can't resolve symbol `tpu_chip_count` with error: %s\n", error_msg);
#endif
      return false;
  }
  _tpu_pids = dlsym(handle, "tpu_pids");
  error_msg = dlerror();
  if (error_msg != NULL) {
#ifndef NDEBUG
      fprintf(stderr, "libtpuinfo can't resolve symbol `tpu_pids` with error: %s\n", error_msg);
#endif
      return false;
  }
  _tpu_metrics = dlsym(handle, "tpu_metrics");
  error_msg = dlerror();
  if (error_msg != NULL) {
#ifndef NDEBUG
      fprintf(stderr, "libtpuinfo can't resolve symbol `tpu_metrics` with error: %s\n", error_msg);
#endif
      return false;
  }

  // Discover TPU devices
  tpu_chip_count = _tpu_chip_count();

  // Allocate memory for local TPU device data cache (if any)
  if (tpu_chip_count > 0) {
    latest_chips_usage_data = (struct tpu_chip_usage_data*)malloc(tpu_chip_count*sizeof(struct tpu_chip_usage_data));
    _pids = (int64*)malloc(sizeof(int64) * tpu_chip_count);
    _device_ids = (int64*)malloc(sizeof(int64) * tpu_chip_count);
    _memory_usage = (int64*)malloc(sizeof(int64) * tpu_chip_count);
    _total_memory = (int64*)malloc(sizeof(int64) * tpu_chip_count);
    _duty_cycle_pct = (double*)malloc(sizeof(double) * tpu_chip_count);
    reset_tpu_cache(true);
  }

  // Load remote hosts if environment variable is set
  load_remote_hosts();

  // Return true if we have local TPUs or remote hosts configured
  if (tpu_chip_count <= 0 && !remote_monitoring_enabled) {
#ifndef NDEBUG
    fprintf(stderr, "Found 0 local TPU devices and no remote hosts configured.\n");
#endif
    return false;
  }

  return true;
}

void free_ptr(void **ptr) {
  if (ptr != NULL && *ptr != NULL) {
    free(*ptr);
    *ptr = NULL;
  }
}

void gpuinfo_tpu_shutdown(void) {
  // Set exit flag to stop any pending operations
  tpu_exit_requested = 1;

  // Wait for running threads to finish
  if (remote_hosts) {
    for (int i = 0; i < remote_host_count; i++) {
      if (remote_hosts[i].thread_running) {
        pthread_join(remote_hosts[i].refresh_thread, NULL);
      }
    }
  }

  free_ptr((void **)&gpu_infos);
  free_ptr((void **)&latest_chips_usage_data);
  free_ptr((void **)&_pids);
  free_ptr((void **)&_device_ids);
  free_ptr((void **)&_memory_usage);
  free_ptr((void **)&_total_memory);
  free_ptr((void **)&_duty_cycle_pct);
  tpu_chip_count = -1;

  // Clean up remote hosts
  if (remote_hosts) {
    for (int i = 0; i < remote_host_count; i++) {
      free(remote_hosts[i].usage_data);
    }
    free(remote_hosts);
    remote_hosts = NULL;
  }
  remote_host_count = 0;
  total_remote_tpus = 0;
  remote_monitoring_enabled = false;
}

const char *gpuinfo_tpu_last_error_string(void) { return "Err"; }

static void add_tpu_chip(struct list_head *devices, unsigned *count,
                         bool is_remote, int host_idx, int remote_dev_id) {
  struct gpu_info_tpu *this_tpu = &gpu_infos[*count];
  this_tpu->base.vendor = &gpu_vendor_tpu;
  this_tpu->device_id = *count;
  this_tpu->is_remote = is_remote;
  this_tpu->remote_host_idx = host_idx;
  this_tpu->remote_device_id = remote_dev_id;

  if (is_remote && host_idx >= 0 && host_idx < remote_host_count) {
    snprintf(this_tpu->base.pdev, PDEV_LEN, "R%d:%d",
             host_idx % 100, remote_dev_id % 100);
  } else {
    snprintf(this_tpu->base.pdev, PDEV_LEN, "TPU%d", remote_dev_id % 100);
  }
  list_add_tail(&this_tpu->base.list, devices);

  this_tpu->base.processes_count = 0;
  this_tpu->base.processes = NULL;
  this_tpu->base.processes_array_size = 0;

  *count = *count + 1;
}

bool gpuinfo_tpu_get_device_handles(struct list_head *devices_list, unsigned *count) {
  *count = 0;
  if (tpu_chip_count <= 0 && !remote_monitoring_enabled) return false;

  // Pre-allocate remote TPUs based on local count (avoid blocking at startup)
  // Actual data will be fetched on first refresh
  int tpus_per_host = (tpu_chip_count > 0) ? tpu_chip_count : 4;
  total_remote_tpus = 0;
  if (remote_monitoring_enabled) {
    for (int i = 0; i < remote_host_count; i++) {
      remote_hosts[i].tpu_count = tpus_per_host;
      if (!remote_hosts[i].usage_data) {
        remote_hosts[i].usage_data = (struct tpu_chip_usage_data *)calloc(
            tpus_per_host, sizeof(struct tpu_chip_usage_data));
      }
      total_remote_tpus += tpus_per_host;
    }
  }

  int total_tpus = tpu_chip_count + total_remote_tpus;
  gpu_infos = (struct gpu_info_tpu *)calloc(total_tpus, sizeof(*gpu_infos));
  if (!gpu_infos) return false;

  // Add local TPUs
  for (int64_t i = 0; i < tpu_chip_count; i++) {
    add_tpu_chip(devices_list, count, false, -1, (int)i);
  }

  // Add remote TPUs
  if (remote_monitoring_enabled) {
    for (int h = 0; h < remote_host_count; h++) {
      for (int d = 0; d < remote_hosts[h].tpu_count; d++) {
        add_tpu_chip(devices_list, count, true, h, d);
      }
    }
  }

  return true;
}

void gpuinfo_tpu_populate_static_info(struct gpu_info *_gpu_info) {
  struct gpu_info_tpu *gpu_info = container_of(_gpu_info, struct gpu_info_tpu, base);
  struct gpuinfo_static_info *static_info = &gpu_info->base.static_info;
  static_info->integrated_graphics = false;
  static_info->encode_decode_shared = false;
  RESET_ALL(static_info->valid);

  const char *model;
  if (gpu_info->is_remote && gpu_info->remote_host_idx >= 0 &&
      gpu_info->remote_host_idx < remote_host_count) {
    // Remote TPU: use model from remote host (will be fetched on first refresh)
    model = remote_hosts[gpu_info->remote_host_idx].tpu_model;
    if (!model[0]) model = "TPU";
    snprintf(static_info->device_name, sizeof(static_info->device_name),
             "%s [%d@%s]", model, gpu_info->remote_device_id,
             remote_hosts[gpu_info->remote_host_idx].ip);
  } else {
    // Local TPU: detect model locally
    model = get_tpu_model_name(gpu_info->remote_device_id);
    snprintf(static_info->device_name, sizeof(static_info->device_name),
             "%s [%d]", model, gpu_info->remote_device_id);
  }
  SET_VALID(gpuinfo_device_name_valid, static_info->valid);
}

void gpuinfo_tpu_refresh_dynamic_info(struct gpu_info *_gpu_info) {
  struct gpu_info_tpu *gpu_info = container_of(_gpu_info, struct gpu_info_tpu, base);
  struct gpuinfo_dynamic_info *dynamic_info = &gpu_info->base.dynamic_info;
  struct gpuinfo_static_info *static_info = &gpu_info->base.static_info;

  struct tpu_chip_usage_data usage_data = {0};

  if (gpu_info->is_remote) {
    // Remote TPU: trigger parallel refresh and get data from cache
    int h = gpu_info->remote_host_idx;
    int d = gpu_info->remote_device_id;
    if (h >= 0 && h < remote_host_count) {
      refresh_all_remote_hosts_parallel();
      if (d >= 0 && d < remote_hosts[h].tpu_count && remote_hosts[h].usage_data) {
        usage_data = remote_hosts[h].usage_data[d];
      }
      // Update device name if model was fetched
      if (remote_hosts[h].tpu_model[0] && strncmp(static_info->device_name, "TPU [", 5) == 0) {
        snprintf(static_info->device_name, sizeof(static_info->device_name),
                 "%s [%d@%s]", remote_hosts[h].tpu_model, d, remote_hosts[h].ip);
      }
    }
  } else {
    // Local TPU: get data from local cache
    refresh_tpu_cache();
    int d = gpu_info->remote_device_id;
    if (d >= 0 && d < tpu_chip_count) {
      usage_data = latest_chips_usage_data[d];
    }
  }

  double mem_util = round(1e2 * (double)(usage_data.memory_usage) / (double)MAX(1, usage_data.total_memory));
  double tpu_util = round(usage_data.duty_cycle_pct);
  SET_GPUINFO_DYNAMIC(dynamic_info, gpu_util_rate, (int)tpu_util);
  SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate, (int)mem_util);
  SET_GPUINFO_DYNAMIC(dynamic_info, total_memory, usage_data.total_memory);
  SET_GPUINFO_DYNAMIC(dynamic_info, used_memory, usage_data.memory_usage);
  SET_GPUINFO_DYNAMIC(dynamic_info, free_memory, usage_data.total_memory - usage_data.memory_usage);
}

void gpuinfo_tpu_get_running_processes(struct gpu_info *_gpu_info) {
  struct gpu_info_tpu *gpu_info = container_of(_gpu_info, struct gpu_info_tpu, base);

  int64_t pid = -1;
  const char *cmdline = NULL;
  const char *user_name = NULL;
  unsigned cpu_percent = 0;
  uint64_t host_mem = 0;

  if (gpu_info->is_remote) {
    // Remote TPU: get process info from remote cache
    int h = gpu_info->remote_host_idx;
    int d = gpu_info->remote_device_id;
    if (h >= 0 && h < remote_host_count &&
        d >= 0 && d < remote_hosts[h].tpu_count &&
        remote_hosts[h].usage_data) {
      pid = remote_hosts[h].usage_data[d].pid;
      cmdline = remote_hosts[h].usage_data[d].cmdline;
      user_name = remote_hosts[h].usage_data[d].user_name;
      cpu_percent = remote_hosts[h].usage_data[d].cpu_percent;
      host_mem = remote_hosts[h].usage_data[d].host_mem;
    }
  } else {
    // Local TPU
    int d = gpu_info->remote_device_id;
    if (d >= 0 && d < tpu_chip_count) {
      pid = latest_chips_usage_data[d].pid;
    }
  }

  if (pid <= 0) {
    _gpu_info->processes_count = 0;
    return;
  }

  _gpu_info->processes_count = 1;
  if (_gpu_info->processes_array_size == 0) {
    _gpu_info->processes_array_size = 1;
    _gpu_info->processes = (struct gpu_process*)malloc(sizeof(struct gpu_process));
    memset(_gpu_info->processes, 0, sizeof(*_gpu_info->processes));
  }
  _gpu_info->processes[0].type = gpu_process_compute;
  _gpu_info->processes[0].pid = pid;
  _gpu_info->processes[0].gpu_memory_usage = _gpu_info->dynamic_info.used_memory;
  SET_VALID(gpuinfo_process_gpu_memory_usage_valid, _gpu_info->processes[0].valid);

  // Set TPU usage (gpu_usage field)
  _gpu_info->processes[0].gpu_usage = _gpu_info->dynamic_info.gpu_util_rate;
  SET_VALID(gpuinfo_process_gpu_usage_valid, _gpu_info->processes[0].valid);

  // Set cmdline for remote TPU processes
  if (cmdline && strlen(cmdline) > 0 && strcmp(cmdline, "N/A") != 0) {
    free(_gpu_info->processes[0].cmdline);
    _gpu_info->processes[0].cmdline = strdup(cmdline);
    SET_VALID(gpuinfo_process_cmdline_valid, _gpu_info->processes[0].valid);
  }

  // Set user_name for remote TPU processes
  if (user_name && strlen(user_name) > 0 && strcmp(user_name, "N/A") != 0) {
    free(_gpu_info->processes[0].user_name);
    _gpu_info->processes[0].user_name = strdup(user_name);
    SET_VALID(gpuinfo_process_user_name_valid, _gpu_info->processes[0].valid);
  }

  // Set CPU and HOST MEM for remote TPU processes
  if (gpu_info->is_remote) {
    _gpu_info->processes[0].cpu_usage = cpu_percent;
    SET_VALID(gpuinfo_process_cpu_usage_valid, _gpu_info->processes[0].valid);
    _gpu_info->processes[0].cpu_memory_res = host_mem;
    SET_VALID(gpuinfo_process_cpu_memory_res_valid, _gpu_info->processes[0].valid);
  }
}
