# RDD2022 inspection (annotated train folders only)

## Label counts (number of boxes)

| Label | Kept? | China_Drone | China_MotorBike | Czech | India | Japan | Norway | United_States | Total |
|---|---|---|---|---|---|---|---|---|---|
| D00 | **yes** | 1426 | 2678 | 988 | 1555 | 4049 | 8570 | 6750 | 26016 |
| D10 | **yes** | 1263 | 1096 | 399 | 68 | 3979 | 1730 | 3295 | 11830 |
| D20 | **yes** | 293 | 641 | 161 | 2021 | 6199 | 468 | 834 | 10617 |
| D40 | **yes** | 86 | 235 | 197 | 3187 | 2243 | 461 | 135 | 6544 |
| D44 | dropped | 0 | 0 | 0 | 1062 | 3995 | 0 | 0 | 5057 |
| D50 | dropped | 0 | 0 | 0 | 28 | 3553 | 0 | 0 | 3581 |
| Repair | dropped | 769 | 277 | 0 | 0 | 0 | 0 | 0 | 1046 |
| D43 | dropped | 0 | 0 | 0 | 57 | 736 | 0 | 0 | 793 |
| D01 | dropped | 0 | 0 | 0 | 179 | 0 | 0 | 0 | 179 |
| D11 | dropped | 0 | 0 | 0 | 45 | 0 | 0 | 0 | 45 |
| Block crack | dropped | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| D0w0 | dropped | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |

Boxes kept: **55007**. Boxes dropped (non-target labels): **10705** (16.3%).

## Images per country

| Country | Images | XMLs | Unpaired | No objects in XML | No target objects | Bad boxes | Most common size |
|---|---|---|---|---|---|---|---|
| China_Drone | 2401 | 2401 | 0 | 5 | 482 | 0 | 512x512 (2401) |
| China_MotorBike | 1977 | 1977 | 0 | 0 | 43 | 0 | 512x512 (1977) |
| Czech | 2829 | 2829 | 0 | 1757 | 1757 | 0 | 600x600 (2829) |
| India | 7706 | 7706 | 0 | 3921 | 4483 | 0 | 720x720 (7706) |
| Japan | 10506 | 10506 | 0 | 794 | 2606 | 1 | 600x600 (10187), 1024x1024 (147) |
| Norway | 8161 | 8161 | 0 | 5247 | 5247 | 0 | 4040x2035 (4342), 3643x2041 (2896) |
| United_States | 4805 | 4805 | 0 | 0 | 0 | 0 | 640x640 (4805) |

*No target objects* = images that end up with an empty label file once non-target labels are dropped.
