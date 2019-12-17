import Classes.progressbar as pb
import time

bar = pb.ProgressBar(max_value=10, widgets=[
    pb.Bar(),
    ' (', pb.ETA(), ') ',
])
for i in range(10):
    time.sleep(0.5)
    bar.update(i)