# Navi-Q: Arduino Nano Quantized TinyML Robot

## Exact original wiring
- L298N IN1 D3, IN2 D4, ENA D5 (left motor)
- L298N IN3 D6, IN4 D7, ENB D9 (right motor)
- HC-SR04 TRIG D8, ECHO D10
- Original motor directions and left/right PWM 195/150 retained.
- Common GND across Nano, L298N, ultrasonic sensor, and battery regulator.
- Remove ENA/ENB jumpers on the L298N if driving those pins by Nano PWM.
- Power motors from a suitable motor battery and Nano from regulated supply. Do not run motors from Nano 5V.

## Flow
1. Nano reads the single fixed forward-facing ultrasonic sensor every ~100 ms.
2. Features: current distance (0..200 cm), positive closing rate (0..100 cm/s), absolute distance change (0..50 cm).
3. Inputs converted to 7-bit integer range 0..127.
4. Custom fixed-point MLP 3->8->3 with signed INT8 weights and biases, Q4 scaling (1/16). Intermediate accumulation INT32; hidden activation uint7; output INT32 argmax.
5. Classes SAFE, CAUTION, DANGER map to normal speed, slow speed, or reverse + fixed left turn.
6. Independent STOP_CM=15 safety threshold, and invalid echo always stops. The model never overrides this.

## Train/deploy
1. In `SONARQ_Nano/SONARQ_Nano.ino`, change `#define DATA_COLLECTION_MODE 0` to `1` and upload using Arduino IDE. Motors stay off.
2. Install Python packages: `pip install pyserial numpy scikit-learn torch`.
3. Run `python collect_data.py COM5 dataset.csv` (Windows) or `python collect_data.py /dev/ttyUSB0 dataset.csv` (Linux).
4. Arrange many different real scenes, including changing distances over time; select the appropriate label for each set of readings. Collect at least 100, preferably 1000+ varied readings. Avoid treating identical repeated samples as independent data.
5. Run `python train_export.py dataset.csv`. This overwrites `SONARQ_Nano/model_weights.h` with trained quantized weights.
6. Set `DATA_COLLECTION_MODE` back to `0` and upload the sketch again. Serial Monitor: 115200 baud.
7. Validate with the wheels lifted, then slow controlled tests in a cleared arena.

## Critical limitations
- The fixed single ultrasonic sensor cannot see side obstacles or choose a verified safe turning direction. The robot uses a fixed left-turn recovery; it may turn into an obstacle.
- Training on hand-labelled scenes teaches the model to imitate those labels; it does NOT establish prediction of actual future collisions. To claim future risk prediction, collect trials with measured future outcomes, an explicit time horizon, and independent held-out scenes.
- Model is quantized INT8 weights/biases with integer-only inference, not a `.tflite` model. It is a custom AVR-friendly network.
- The training script uses a preliminary stratified random split; for credible performance evaluation, hold out entire obstacle setups and sessions.
- This is a prototype; use low speeds and a clear test area. No person/pet safety use.
- Motor recovery timings are open-loop, not guaranteed. Adjust experimentally.
