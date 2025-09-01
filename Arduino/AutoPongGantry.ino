// AutoPongGantry.ino
// Minimal 2-axis STEP/DIR controller that accepts simple serial commands.
// Requires AccelStepper library.

#include <AccelStepper.h>

/////////////////////// CONFIG ///////////////////////
#define PIN_X_STEP 2
#define PIN_X_DIR  5
#define PIN_Y_STEP 3
#define PIN_Y_DIR  6

// steps per mm (set from your mechanics: microsteps * steps_per_rev / mm_per_rev)
const float STEPS_PER_MM_X = 80.0f;
const float STEPS_PER_MM_Y = 80.0f;

// motion limits (mm/s and mm/s^2)
float MAX_V_NORMAL = 300.0f;
float MAX_A_NORMAL = 1500.0f;
float MAX_V_FAST   = 600.0f;   // used only when mode == SMART (U)
float MAX_A_FAST   = 4000.0f;

const unsigned long STATUS_PERIOD_MS = 1000;

//////////////////////////////////////////////////////

AccelStepper stepX(AccelStepper::DRIVER, PIN_X_STEP, PIN_X_DIR);
AccelStepper stepY(AccelStepper::DRIVER, PIN_Y_STEP, PIN_Y_DIR);

enum Mode { MODE_IDLE, MODE_TARGET, MODE_SMART, MODE_STOP, MODE_REHOME };
Mode mode = MODE_IDLE;

float tgt_x_mm = 0.0f, tgt_y_mm = 0.0f;
unsigned long lastStatus = 0;

String rx;

void setLimitsForMode() {
  if (mode == MODE_SMART) {
    stepX.setMaxSpeed(MAX_V_FAST * STEPS_PER_MM_X);
    stepY.setMaxSpeed(MAX_V_FAST * STEPS_PER_MM_Y);
    stepX.setAcceleration(MAX_A_FAST * STEPS_PER_MM_X);
    stepY.setAcceleration(MAX_A_FAST * STEPS_PER_MM_Y);
  } else {
    stepX.setMaxSpeed(MAX_V_NORMAL * STEPS_PER_MM_X);
    stepY.setMaxSpeed(MAX_V_NORMAL * STEPS_PER_MM_Y);
    stepX.setAcceleration(MAX_A_NORMAL * STEPS_PER_MM_X);
    stepY.setAcceleration(MAX_A_NORMAL * STEPS_PER_MM_Y);
  }
}

long mm2stepsX(float mm) { return lroundf(mm * STEPS_PER_MM_X); }
long mm2stepsY(float mm) { return lroundf(mm * STEPS_PER_MM_Y); }
float steps2mmX(long st) { return (float)st / STEPS_PER_MM_X; }
float steps2mmY(long st) { return (float)st / STEPS_PER_MM_Y; }

void sendStatus() {
  float x = steps2mmX(stepX.currentPosition());
  float y = steps2mmY(stepY.currentPosition());
  char m = (mode == MODE_IDLE) ? 'I' :
           (mode == MODE_TARGET) ? 'T' :
           (mode == MODE_SMART) ? 'U' :
           (mode == MODE_STOP) ? 'X' : 'R';
  Serial.print("POS,"); Serial.print(x, 3);
  Serial.print(",");     Serial.print(y, 3);
  Serial.print(",");     Serial.println(m);
}

void gotoTarget(float x_mm, float y_mm) {
  tgt_x_mm = x_mm; tgt_y_mm = y_mm;
  stepX.moveTo(mm2stepsX(tgt_x_mm));
  stepY.moveTo(mm2stepsY(tgt_y_mm));
}

void stopNow() {
  // Halt aggressively: set target to current pos and zero speed quickly
  stepX.stop(); stepY.stop();
  stepX.setCurrentPosition(stepX.currentPosition());
  stepY.setCurrentPosition(stepY.currentPosition());
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  // Split by commas
  const int MAXP = 8;
  String parts[MAXP];
  int n = 0;
  int from = 0;
  while (n < MAXP) {
    int idx = line.indexOf(',', from);
    if (idx < 0) { parts[n++] = line.substring(from); break; }
    parts[n++] = line.substring(from, idx);
    from = idx + 1;
  }
  parts[0].toUpperCase();
  String cmd = parts[0];

  if (cmd == "I") {
    mode = MODE_IDLE;
    setLimitsForMode();
  } else if (cmd == "X" || cmd == "STOP") {
    mode = MODE_STOP;
    stopNow();
  } else if (cmd == "R") {
    mode = MODE_REHOME;
    // TODO: implement homing to switches here
    // For now, just zero the encoders/positions.
    stepX.setCurrentPosition(0);
    stepY.setCurrentPosition(0);
    mode = MODE_IDLE;
  } else if (cmd == "T") {
    if (n >= 3) {
      float x = parts[1].toFloat();
      float y = parts[2].toFloat();
      if (mode == MODE_IDLE) mode = MODE_TARGET;
      setLimitsForMode();
      gotoTarget(x, y);
    }
  } else if (cmd == "U") {
    // Smart mode: same target semantics; Python does planning.
    if (n >= 3) {
      float x = parts[1].toFloat();
      float y = parts[2].toFloat();
      mode = MODE_SMART;
      setLimitsForMode();
      gotoTarget(x, y);
      // vx,vy in parts[3],parts[4] are accepted but not used here.
    }
  } else if (cmd == "P" && n >= 3) {
    // Parameter set: P,KEY,VAL
    String key = parts[1]; key.toUpperCase();
    float val = parts[2].toFloat();
    if      (key == "MAX_V")      MAX_V_NORMAL = val;
    else if (key == "MAX_A")      MAX_A_NORMAL = val;
    else if (key == "MAX_V_FAST") MAX_V_FAST   = val;
    else if (key == "MAX_A_FAST") MAX_A_FAST   = val;
    setLimitsForMode();
    Serial.println("ACK");
  } else if (cmd == "Q") {
    sendStatus();
  } else {
    Serial.println("ERR,unknown_cmd");
  }
}

void setup() {
  Serial.begin(115200);
  stepX.setEnablePin(-1);
  stepY.setEnablePin(-1);
  setLimitsForMode();
  sendStatus();
}

void loop() {
  // Parse serial lines
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rx.length() > 0) { handleLine(rx); rx = ""; }
    } else {
      rx += c;
      if (rx.length() > 200) rx = ""; // basic flood protection
    }
  }

  // Motion
  stepX.run();
  stepY.run();

  // Periodic status
  unsigned long now = millis();
  if (now - lastStatus >= STATUS_PERIOD_MS) {
    lastStatus = now;
    sendStatus();
  }
}
