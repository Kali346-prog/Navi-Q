#include <Arduino.h>
#include <avr/pgmspace.h>
#include "model_weights.h"

// Original pin assignments and motor directions preserved.
const uint8_t IN1=3, IN2=4, ENA=5, IN3=6, IN4=7, ENB=9;
const uint8_t TRIG=8, ECHO=10;
const uint8_t LEFT_PWM=195, RIGHT_PWM=150;

// Set to 1 when collecting labelled training data; motors stay stopped.
#define DATA_COLLECTION_MODE 0

const uint16_t SAMPLE_MS=100;
const uint16_t ECHO_TIMEOUT_US=12000; // ~206 cm max
const uint8_t STOP_CM=15; // independent safety threshold
const uint8_t CAUTION_CM=35;
const uint16_t REVERSE_MS=400;
const uint16_t TURN_MS=500;
const uint16_t SETTLE_MS=120;

uint32_t lastSample=0, lastValidMs=0;
int16_t lastDistance=-1;
uint8_t invalidCount=0;
enum State {DRIVE, BACKUP, TURN, SETTLE};
State state=DRIVE;
uint32_t stateUntil=0;

void stopMotors() {
  digitalWrite(IN1,LOW); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,LOW);
  analogWrite(ENA,0); analogWrite(ENB,0);
}
void forward(uint8_t l=LEFT_PWM, uint8_t r=RIGHT_PWM) {
  digitalWrite(IN1,HIGH); digitalWrite(IN2,LOW);
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH);
  analogWrite(ENA,l); analogWrite(ENB,r);
}
void backward() {
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH);
  digitalWrite(IN3,HIGH); digitalWrite(IN4,LOW);
  analogWrite(ENA,LEFT_PWM); analogWrite(ENB,RIGHT_PWM);
}
void turnLeft() {
  digitalWrite(IN1,LOW); digitalWrite(IN2,HIGH);
  digitalWrite(IN3,LOW); digitalWrite(IN4,HIGH);
  analogWrite(ENA,LEFT_PWM); analogWrite(ENB,RIGHT_PWM);
}
// Returns -1 for missing echo, or measured distance in cm.
int16_t getDistance() {
  digitalWrite(TRIG,LOW); delayMicroseconds(2);
  digitalWrite(TRIG,HIGH); delayMicroseconds(10);
  digitalWrite(TRIG,LOW);
  unsigned long us=pulseIn(ECHO,HIGH,ECHO_TIMEOUT_US);
  if (!us) return -1;
  return (int16_t)((us*343UL)/20000UL);
}
uint8_t toQ(int16_t v, int16_t maxV) {
  if (v<0) v=0;
  if (v>maxV) v=maxV;
  return (uint8_t)(((int32_t)v*127 + maxV/2)/maxV);
}
// Model: input uint8 [0..127], weights int8 with scale 1/16,
// biases int8 with scale 1/16. Hidden ReLU quantized to [0..127].
// Output is argmax over integer logits (no softmax necessary).
uint8_t predictRisk(const uint8_t x[3]) {
  int16_t h[8];
  for (uint8_t j=0;j<8;j++) {
    int32_t acc=(int32_t)(int8_t)pgm_read_byte(&B1[j])*127;
    for (uint8_t i=0;i<3;i++)
      acc+=(int32_t)x[i]*(int8_t)pgm_read_byte(&W1[j][i]);
    // Quantization scale 1/16; floor rounding acceptable and reproduced in Python.
    int32_t val=acc/16;
    if (val<0) val=0;
    if (val>127) val=127;
    h[j]=(int16_t)val;
  }
  int32_t best=-2147483647L;
  uint8_t cls=2; // 0 safe, 1 caution, 2 danger
  for (uint8_t k=0;k<3;k++) {
    int32_t acc=(int32_t)(int8_t)pgm_read_byte(&B2[k])*127;
    for (uint8_t j=0;j<8;j++)
      acc+=(int32_t)h[j]*(int8_t)pgm_read_byte(&W2[k][j]);
    if (acc>best) { best=acc; cls=k; }
  }
  return cls;
}
void beginAvoidance(uint32_t now) {
  stopMotors();
  state=BACKUP;
  stateUntil=now+REVERSE_MS;
  backward();
}
void setup() {
  pinMode(IN1,OUTPUT);pinMode(IN2,OUTPUT);pinMode(ENA,OUTPUT);
  pinMode(IN3,OUTPUT);pinMode(IN4,OUTPUT);pinMode(ENB,OUTPUT);
  pinMode(TRIG,OUTPUT);pinMode(ECHO,INPUT);
  stopMotors();
  Serial.begin(115200);
  Serial.println(F("SONARQ Nano: dist_cm,closing_cm_s,delta_cm,risk"));
#if !MODEL_TRAINED
  Serial.println(F("WARNING: No trained model. Conservative rule baseline only."));
#endif
#if DATA_COLLECTION_MODE
  Serial.println(F("DATA COLLECTION MODE: motors disabled"));
#endif
}
void loop() {
  uint32_t now=millis();
#if !DATA_COLLECTION_MODE
  if (state!=DRIVE) {
    if ((int32_t)(now-stateUntil)<0) return;
    if (state==BACKUP) {stopMotors();turnLeft();state=TURN;stateUntil=now+TURN_MS;return;}
    if (state==TURN) {stopMotors();state=SETTLE;stateUntil=now+SETTLE_MS;return;}
    state=DRIVE;
    lastDistance=-1; // reset temporal history after maneuver
    stopMotors();
  }
#endif
  if ((uint32_t)(now-lastSample)<SAMPLE_MS) return;
  uint32_t elapsed=(lastSample==0)?SAMPLE_MS:(uint32_t)(now-lastSample);
  lastSample=now;
  int16_t d=getDistance();
  if (d<0) {
    invalidCount++;
    stopMotors();
    lastDistance=-1;
    Serial.println(F("INVALID_ECHO: STOP"));
    return;
  }
  invalidCount=0;
  lastValidMs=now;
  int16_t delta=(lastDistance<0)?0:(lastDistance-d); // positive when closing
  int16_t closing=(int16_t)((int32_t)delta*1000L/(int32_t)elapsed);
  if (closing<0) closing=0;
  if (closing>100) closing=100;
  int16_t absDelta=delta<0?-delta:delta;
  if (absDelta>50) absDelta=50;
  lastDistance=d;
  uint8_t x[3]={toQ(d,200),toQ(closing,100),toQ(absDelta,50)};
  uint8_t risk=2;
#if MODEL_TRAINED
  risk=predictRisk(x);
#else
  risk=(d<=STOP_CM)?2:((d<=CAUTION_CM)?1:0);
#endif
  Serial.print(d);Serial.print(',');Serial.print(closing);
  Serial.print(',');Serial.print(absDelta);Serial.print(',');Serial.println(risk);
#if DATA_COLLECTION_MODE
  stopMotors();
#else
  if (d<=STOP_CM || risk==2) {
    beginAvoidance(now);
  } else if (d<=CAUTION_CM || risk==1) {
    forward(120,95);
  } else {
    forward();
  }
#endif
}
