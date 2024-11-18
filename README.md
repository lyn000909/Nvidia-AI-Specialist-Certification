![NVIDIA LOGO](https://github.com/user-attachments/assets/9cf87f01-ff75-4c6a-b4c8-2560ca2e4db7)

# Nvidia AI Specialist Certification
### <span style="color:violet">Title : Vehicle license plate recognition system using yolov5</span>
---
## ✅ OverView of the Project
    - Opening background information

    - General description of the current project

    - Proposed idea for enhancements to the project

    - Value and significance of this project

    - Current limitations

    - Literature review
---
## Opening background information
```
✅ Recent developments in autonomous driving technology have attracted more attention with
artificial intelligence-based recognition systems combining LiDAR and cameras, and these technologies
are playing an important role in a range of applications such as size plate recognition for vehicles
(automatic license plate recognition, ALPR).
```
---
## General description of the current project
```
✅ This project is to develop a vehicle license plate recognition system using YOLOv5.
It recognizes the vehicle's license plate as a partner, acknowledges it, and performs various functions
such as tracking stolen vehicles and automatic parking management.
```
---
## Proposed idea for enhancements to the project
```
✅ YOLOv5 processes the entire image at once to achieve accurate object detection while maintaining
high frame rates. This allows license plate recognition in real time even in environments where
vehicles are moving quickly.
```
---
## Value and significance of this project
```
✅ This project can be used in various fields such as traffic management, law enforcement,
and parking management through vehicle license plate recognition, and is an essential element
for autonomous vehicles to comply with traffic rules in real time and identify violative vehicles.
```
---
## Current limitations
```
✅ Limitations faced include poor recognition performance in various environmental conditions.
For example, when driving in bad weather or at night, it is difficult to accurately recognize license plates
using cameras alone, and vehicles moving at high speeds or partially obscured license plates
are highly likely to cause errors in existing systems.
```
---
## Literature review
```
✅ Because there are limitations such as dependency issues on large-scale datasets and poor performance
in various environments, improvements are needed by securing more datasets and using additional sensor
fusion technologies such as LiDAR.
```
---
## <span style="color:blue"> Image Acquisition Method </span>
- The parking lot black box video was filmed, and additional video
  was referenced from the Seoul road test driving video.

[DRIVE TEST](https://github.com/user-attachments/assets/9bfaefa1-c508-4fa7-a04f-94441b3b1514)

## <span style="color:blue">Learning Data Extraction and Learning Annotation </span>

- In order to learn with 640 x 640 resolution images in YOLOv5,

  the images were first created as 640 x 640 resolution images.

## <span style="color:blue"> Video Resolution Adjustment </span>
<https://online-video-cutter.com/ko/resize-video>

![VIDEO RESIZER](https://github.com/user-attachments/assets/ebb7d188-355c-47f3-91e6-61c72c12c911)

- I used Darklabel to create edits with images based on frames, due to 640 x 640.

[DarkLabel2.4.zip](https://github.com/user-attachments/files/17794875/DarkLabel2.4.zip)

![darklabel 5](https://github.com/user-attachments/assets/1769e2b0-84ba-4854-beaa-2e4dd4cecf4c)

- First, add classes through darklabel.yml before annotation.

![code](https://github.com/user-attachments/assets/6cef410f-bcf9-4d95-b729-8cec67256bfc)

- Add vehicle classes in the yaml file and add vehicle license plate as the class name.

![code 1](https://github.com/user-attachments/assets/d663c898-80e5-4a71-b6ec-68a3a8a85911)

- When annotating, put the vehicle classes in classes_set so that you can see the classes set

  in the DarkLabel GUI, and set the name to be viewed in the GUI to vehicle license plate.

![darklabel 2](https://github.com/user-attachments/assets/fb3cb492-bf57-4077-b8c3-864958c4b68e)

- You can see that classes called vehicle classes have been added to the DarkLabel program,

  and a vehicle license plate has been added below.

  




