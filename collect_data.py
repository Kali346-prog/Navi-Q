"""Collect labelled readings with the robot STATIONARY in DATA_COLLECTION_MODE=1.
Usage: python collect_data.py COM5 dataset.csv  (or /dev/ttyUSB0)
Label scenes yourself. This is behavioral imitation, not proven future collision prediction.
"""
import serial,csv,sys,time,pathlib
if len(sys.argv)!=3:raise SystemExit('Usage: python collect_data.py PORT dataset.csv')
port,out=sys.argv[1:]
ser=serial.Serial(port,115200,timeout=2)
time.sleep(2)
p=pathlib.Path(out)
with p.open('a',newline='') as f:
    writer=csv.writer(f)
    if p.stat().st_size==0:writer.writerow(['distance_cm','closing_cm_s','delta_cm','label'])
    while True:
        label=input('Arrange a scene. Enter SAFE / CAUTION / DANGER or Q: ').strip().upper()
        if label=='Q':break
        if label not in ('SAFE','CAUTION','DANGER'):continue
        n=int(input('Number of readings to capture (e.g. 40): '))
        ser.reset_input_buffer()
        count=0
        while count<n:
            line=ser.readline().decode('ascii',errors='ignore').strip().split(',')
            if len(line)!=4:continue
            try:
                d,c,delta=[int(v) for v in line[:3]]
            except ValueError:continue
            writer.writerow([d,c,delta,label]);count+=1
        f.flush();print('Saved',count,'samples as',label)
