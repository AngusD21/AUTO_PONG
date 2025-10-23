// AutoPongGantry.ino
// Minimal 2-axis STEP/DIR controller that accepts simple serial commands.
// Requires AccelStepper library.

#include <AccelStepper.h>

/////////////////////// HARDWARE MAP ///////////////////////
// Motor 1 (X): DM556  PUL- -> D5 , DIR- -> D4  (PUL+/DIR+ tied to +5V)
#define PIN_X_STEP 5
#define PIN_X_DIR  4

// Motor 2 (Y): DM556  PUL- -> D6 , DIR- -> D7  (PUL+/DIR+ tied to +5V)
#define PIN_Y_STEP 6
#define PIN_Y_DIR  7

// E-Stop (NC recommended): switch to GND, uses INPUT_PULLUP
#define PIN_ESTOP  8

// Logic polarity (typical for PUL+/DIR+ tied to +5V)
const bool STEP_ACTIVE_LOW = true;   // DM556 opto turns on when PUL- is LOW
const bool X_DIR_INVERT    = false;  // flip if X runs the wrong way
const bool Y_DIR_INVERT    = false;  // flip if Y runs the wrong way
///////////////////////////////////////////////////////////

/////////////////////// KINEMATICS /////////////////////////
// steps per mm (microsteps * steps_per_rev / mm_per_rev)
const float STEPS_PER_MM_X = 80.0f;
const float STEPS_PER_MM_Y = 80.0f;

/////////////////////// LIMITS /////////////////////////////
float MAX_V_NORMAL = 300.0f;   // mm/s
float MAX_A_NORMAL = 1500.0f;  // mm/s^2
float MAX_V_FAST   = 600.0f;   // SMART (U) only
float MAX_A_FAST   = 4000.0f;

const unsigned long STATUS_PERIOD_MS = 1000;
///////////////////////////////////////////////////////////

AccelStepper stepX(AccelStepper::DRIVER, PIN_X_STEP, PIN_X_DIR);
AccelStepper stepY(AccelStepper::DRIVER, PIN_Y_STEP, PIN_Y_DIR);

enum Mode { MODE_IDLE, MODE_TARGET, MODE_SMART, MODE_STOP, MODE_REHOME };
Mode mode = MODE_IDLE;

float tgt_x_mm = 0.0f, tgt_y_mm = 0.0f;
unsigned long lastStatus = 0;
String rx;

volatile bool estopActive = false;   // live input state
bool estopLatched = false;           // latched state while pressed

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
  char m = (mode == MODE_IDLE)   ? 'I' :
           (mode == MODE_TARGET) ? 'T' :
           (mode == MODE_SMART)  ? 'U' :
           (mode == MODE_STOP)   ? 'X' : 'R';
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
  // Halt quickly and hold
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
  int n = 0, from = 0;
  while (n < MAXP) {
    int idx = line.indexOf(',', from);
    if (idx < 0) { parts[n++] = line.substring(from); break; }
    parts[n++] = line.substring(from, idx);
    from = idx + 1;
  }
  parts[0].toUpperCase();
  String cmd = parts[0];

  // If E-STOP is latched, ignore everything except query
  if (estopLatched) {
    if (cmd == "Q") { sendStatus(); return; }
    Serial.println("ERR,estop_active");
    return;
  }

  if (cmd == "I") {
    mode = MODE_IDLE;
    setLimitsForMode();
  } else if (cmd == "X" || cmd == "STOP") {
    mode = MODE_STOP;
    stopNow();
  } else if (cmd == "R") {
    mode = MODE_REHOME;
    // TODO: implement homing inputs; for now, zero positions
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
    // SMART mode: Python plans; Arduino tracks target points
    if (n >= 3) {
      float x = parts[1].toFloat();
      float y = parts[2].toFloat();
      mode = MODE_SMART;
      setLimitsForMode();
      gotoTarget(x, y);
      // vx,vy (parts[3],parts[4]) are accepted but not used here
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

  // E-Stop as normally-closed to GND (pressed -> LOW)
  pinMode(PIN_ESTOP, INPUT_PULLUP);

  // Configure step/dir polarity
  // setPinsInverted(directionInvert, stepInvert, enableInvert)
  stepX.setPinsInverted(X_DIR_INVERT, STEP_ACTIVE_LOW, false);
  stepY.setPinsInverted(Y_DIR_INVERT, STEP_ACTIVE_LOW, false);

  // (Optional) enable pins not used in this wiring
  stepX.setEnablePin(-1);
  stepY.setEnablePin(-1);

  setLimitsForMode();
  sendStatus();
}

void loop() {
  // --- E-STOP handling (edge detect & latch while pressed) ---
  bool pressed = (digitalRead(PIN_ESTOP) == LOW);
  if (pressed && !estopLatched) {
    estopLatched = true;
    stopNow();
    stepX.disableOutputs();
    stepY.disableOutputs();
    mode = MODE_STOP;
    Serial.println("ESTOP,1");
  } else if (!pressed && estopLatched) {
    // Released -> allow commands again; remain IDLE until told otherwise
    estopLatched = false;
    stepX.enableOutputs();
    mode = MODE_IDLE;
    Serial.println("ESTOP,0");
    sendStatus();
  }

  // --- Serial parsing ---
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (rx.length() > 0) { handleLine(rx); rx = ""; }
    } else {
      rx += c;
      if (rx.length() > 200) rx = ""; // basic flood protection
    }
  }

  // --- Motion (no motion while estopLatched) ---
  if (!estopLatched) {
    stepX.run();
    stepY.run();
  }

  // --- Periodic status ---
  unsigned long now = millis();
  if (now - lastStatus >= STATUS_PERIOD_MS) {
    lastStatus = now;
    sendStatus();
  }
}
