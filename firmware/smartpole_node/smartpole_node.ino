/*
  SmartPole node firmware
  ESP32-CAM (AI-Thinker layout) with a GC2145 camera sensor.

  The node works on its own: it captures a photo every CAPTURE_INTERVAL_MS
  and PUSHES it to the laptop server over Wi-Fi. The laptop never asks the
  camera for anything.

  Arduino IDE  ->  Tools menu:
    Board            : ESP32 Dev Module   (exposes the PSRAM option)
    PSRAM            : Enabled
    Partition Scheme : Huge APP (3MB No OTA/1MB SPIFFS)
    Upload Speed     : 115200             (921600 failed to verify flash)
    Port             : the COM port of the ESP32-CAM-MB board (e.g. COM9)
  Tested against arduino-esp32 core 3.3.12.

  Wi-Fi name and password live in secrets.h (gitignored, see secrets.example.h).
*/

#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#include "img_converters.h"   // frame2jpg(): software JPEG encoder

#if __has_include("secrets.h")
#include "secrets.h"
#else
#error "Copy secrets.example.h to secrets.h and fill in your Wi-Fi name and password."
#endif

// ======================= Settings: change these =======================

const char *POLE_ID = "POLE01";          // sent with every upload

// Used only if UDP discovery finds nothing. Put the laptop's hotspot IP
// here (run `ipconfig` on the laptop, look under "Wireless LAN adapter Wi-Fi").
const char *FALLBACK_SERVER_IP = "192.168.43.100";
const uint16_t FALLBACK_SERVER_PORT = 8000;

const uint32_t CAPTURE_INTERVAL_MS   = 15000;   // one photo every 15 s
const uint32_t HEARTBEAT_INTERVAL_MS = 10000;   // "I am alive" every 10 s

// Start with QVGA (320x240). Once uploads are stable, try FRAMESIZE_VGA
// (640x480) and FRAMESIZE_SVGA (800x600).
const framesize_t FRAME_SIZE = FRAMESIZE_QVGA;
const uint8_t JPEG_QUALITY   = 80;     // 1-100 for frame2jpg, higher = better

// If images come out corrupted or striped, try 10000000 (10 MHz).
const int XCLK_FREQ_HZ = 20000000;

// UDP discovery: the server broadcasts "SMARTPOLE_SERVER <port>" on this
// port every 2 s. Must match DISCOVERY_PORT in config/settings.py.
const uint16_t DISCOVERY_PORT = 50000;
const uint32_t DISCOVERY_WAIT_MS = 10000;

// After this many failed uploads in a row, search for the server again
// (the laptop's IP may have changed).
const int MAX_UPLOAD_FAILURES = 3;

// ================== AI-Thinker pin map ==================
// Checked against CAMERA_MODEL_AI_THINKER in the core's camera_pins.h.
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM    5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22
#define FLASH_LED_GPIO 4

// ======================= Global state =======================

String serverIp = "";
uint16_t serverPort = 0;
uint32_t seq = 0;                 // counts captures since boot
int uploadFailures = 0;
int cameraFailures = 0;
uint32_t lastCaptureMs = 0;
uint32_t lastHeartbeatMs = 0;

// ======================= Camera =======================

bool initCamera() {
  camera_config_t config = {};
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.xclk_freq_hz = XCLK_FREQ_HZ;
  config.ledc_timer   = LEDC_TIMER_0;
  config.ledc_channel = LEDC_CHANNEL_0;

  // The GC2145 has no hardware JPEG encoder. Asking for PIXFORMAT_JPEG
  // fails with error 0x106 (ESP_ERR_NOT_SUPPORTED), so we take raw RGB565
  // pixels and convert them to JPEG in software with frame2jpg().
  config.pixel_format = PIXFORMAT_RGB565;
  config.frame_size   = FRAME_SIZE;
  config.jpeg_quality = 12;                 // unused for RGB565
  config.fb_count     = 1;
  config.fb_location  = CAMERA_FB_IN_PSRAM; // raw frames are too big for internal RAM
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init FAILED, error 0x%x\n", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  Serial.printf("[CAM] init OK, sensor PID 0x%x\n", s ? s->id.PID : 0);

  // The first frames after start-up are often dark or garbled. Throw them away.
  for (int i = 0; i < 2; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(100);
  }
  return true;
}

// Captures one frame and converts it to JPEG.
// On success the caller owns *jpgBuf and MUST free() it.
bool captureJpeg(uint8_t **jpgBuf, size_t *jpgLen) {
  // With one frame buffer and GRAB_WHEN_EMPTY, the buffer was filled right
  // after the previous capture, i.e. it is ~15 s old. Drop it to get a fresh one.
  camera_fb_t *fb = esp_camera_fb_get();
  if (fb) esp_camera_fb_return(fb);

  fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[CAM] capture FAILED (no frame)");
    return false;
  }

  bool ok = frame2jpg(fb, JPEG_QUALITY, jpgBuf, jpgLen);
  int w = fb->width, h = fb->height;
  esp_camera_fb_return(fb);   // give the raw buffer back right away

  if (!ok) {
    Serial.println("[CAM] JPEG conversion FAILED");
    return false;
  }
  Serial.printf("[CAM] captured %dx%d -> %u byte JPEG\n", w, h, (unsigned)*jpgLen);
  return true;
}

// ======================= Wi-Fi =======================

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.printf("[WIFI] connecting to \"%s\"", WIFI_SSID);
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("[WIFI] connected, IP %s, signal %d dBm\n",
                  WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else {
    // Most common causes: hotspot set to 5 GHz, wrong password, out of range.
    Serial.println("[WIFI] not connected yet (is the hotspot on 2.4 GHz?)");
  }
}

// ======================= Server discovery =======================

// Listens for the server's UDP broadcast. The packet's sender address is
// the laptop's current IP, so the node follows the laptop if its IP changes.
bool discoverServer() {
  WiFiUDP udp;
  udp.begin(DISCOVERY_PORT);
  Serial.printf("[DISC] listening for server on UDP %u for %u s...\n",
                DISCOVERY_PORT, (unsigned)(DISCOVERY_WAIT_MS / 1000));

  bool found = false;
  uint32_t start = millis();
  while (millis() - start < DISCOVERY_WAIT_MS) {
    if (udp.parsePacket() > 0) {
      char msg[64] = {0};
      udp.read(msg, sizeof(msg) - 1);
      unsigned int port = 0;
      if (sscanf(msg, "SMARTPOLE_SERVER %u", &port) == 1 && port > 0) {
        serverIp = udp.remoteIP().toString();
        serverPort = port;
        found = true;
        break;
      }
    }
    delay(50);
  }
  udp.stop();

  if (found) {
    Serial.printf("[DISC] server found at %s:%u\n", serverIp.c_str(), serverPort);
  } else {
    serverIp = FALLBACK_SERVER_IP;
    serverPort = FALLBACK_SERVER_PORT;
    Serial.printf("[DISC] no broadcast heard, using fallback %s:%u\n",
                  serverIp.c_str(), serverPort);
  }
  return found;
}

// ======================= HTTP =======================

String serverUrl(const char *path) {
  return "http://" + serverIp + ":" + String(serverPort) + path;
}

bool uploadJpeg(uint8_t *jpg, size_t len) {
  WiFiClient client;
  HTTPClient http;
  http.setConnectTimeout(5000);
  http.setTimeout(10000);
  if (!http.begin(client, serverUrl("/upload"))) {
    Serial.println("[HTTP] upload: could not start request");
    return false;
  }
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Pole-ID", POLE_ID);
  http.addHeader("X-Seq", String(seq));

  int code = http.POST(jpg, len);
  http.end();

  if (code == 200) {
    Serial.printf("[HTTP] upload OK  seq=%u  %u bytes  HTTP %d\n",
                  (unsigned)seq, (unsigned)len, code);
    return true;
  }
  if (code > 0) {
    Serial.printf("[HTTP] upload FAILED  seq=%u  HTTP %d\n", (unsigned)seq, code);
  } else {
    // Negative codes are connection errors, e.g. "connection refused" when
    // the server is not running or the Windows Firewall blocks port 8000.
    Serial.printf("[HTTP] upload FAILED  seq=%u  %s\n", (unsigned)seq,
                  HTTPClient::errorToString(code).c_str());
  }
  return false;
}

void sendHeartbeat() {
  char body[256];
  snprintf(body, sizeof(body),
           "{\"pole_id\":\"%s\",\"uptime_s\":%lu,\"rssi\":%d,"
           "\"free_heap\":%u,\"free_psram\":%u,\"seq\":%u,\"ip\":\"%s\"}",
           POLE_ID, (unsigned long)(millis() / 1000), WiFi.RSSI(),
           (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram(),
           (unsigned)seq, WiFi.localIP().toString().c_str());

  WiFiClient client;
  HTTPClient http;
  http.setConnectTimeout(3000);
  http.setTimeout(5000);
  if (!http.begin(client, serverUrl("/heartbeat"))) return;
  http.addHeader("Content-Type", "application/json");
  http.addHeader("X-Pole-ID", POLE_ID);
  int code = http.POST((uint8_t *)body, strlen(body));
  http.end();

  if (code == 200) {
    Serial.printf("[HB] ok  free heap %u, free PSRAM %u\n",
                  (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
  } else {
    Serial.printf("[HB] failed: %s\n",
                  code > 0 ? String(code).c_str() : HTTPClient::errorToString(code).c_str());
  }
}

// ======================= Main =======================

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== SmartPole node starting ===");

  pinMode(FLASH_LED_GPIO, OUTPUT);
  digitalWrite(FLASH_LED_GPIO, LOW);   // keep the bright flash LED off

  Serial.printf("[SYS] PSRAM found: %s, free PSRAM %u bytes\n",
                psramFound() ? "yes" : "NO (check Tools > PSRAM)",
                (unsigned)ESP.getFreePsram());

  if (!initCamera()) {
    Serial.println("[SYS] camera failed, restarting in 5 s");
    delay(5000);
    ESP.restart();
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);          // Wi-Fi power saving causes slow/failed uploads
  WiFi.setAutoReconnect(true);
  connectWiFi();
  if (WiFi.status() == WL_CONNECTED) discoverServer();

  // Make the first capture and heartbeat happen straight away.
  lastCaptureMs = millis() - CAPTURE_INTERVAL_MS;
  lastHeartbeatMs = millis() - HEARTBEAT_INTERVAL_MS;
}

void loop() {
  // 1. Stay on Wi-Fi.
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WIFI] connection lost");
    connectWiFi();
    if (WiFi.status() != WL_CONNECTED) {
      delay(5000);
      return;
    }
    discoverServer();   // the laptop may have a new IP after a reconnect
  }
  if (serverIp == "") discoverServer();

  uint32_t now = millis();

  // 2. Heartbeat, so the dashboard can show whether the camera is online.
  if (now - lastHeartbeatMs >= HEARTBEAT_INTERVAL_MS) {
    lastHeartbeatMs = now;
    sendHeartbeat();
  }

  // 3. Capture and upload.
  if (now - lastCaptureMs >= CAPTURE_INTERVAL_MS) {
    lastCaptureMs = now;
    seq++;

    uint8_t *jpg = nullptr;
    size_t jpgLen = 0;
    if (captureJpeg(&jpg, &jpgLen)) {
      cameraFailures = 0;
      bool ok = uploadJpeg(jpg, jpgLen);
      uploadFailures = ok ? 0 : uploadFailures + 1;
    } else {
      cameraFailures++;
    }
    free(jpg);   // frame2jpg allocated it; free(nullptr) is safe if capture failed

    if (uploadFailures >= MAX_UPLOAD_FAILURES) {
      Serial.println("[DISC] several uploads failed, searching for server again");
      uploadFailures = 0;
      discoverServer();
    }
    if (cameraFailures >= 5) {
      Serial.println("[SYS] camera keeps failing, restarting");
      delay(1000);
      ESP.restart();
    }
  }

  delay(20);
}
