## This project has been discontinued

Yes, it still works, you can still use this software. It just won't recieve any updates now.

> I do not have the interest or time to oversee the development of this software. I thank all the amazing people who contributed to this project and made what it is in it's final form.

# Roop项目UI改进技术文档

## 1. 项目概述

Roop是一个基于深度学习的换脸项目，通过使用insightface和GFPGAN等深度学习模型实现高质量的人脸替换功能。本文档主要介绍对原项目UI界面的改进，包括新增的人脸库功能和实时换脸功能。

## 2. UI改进内容

### 2.1 界面布局优化

- 采用CustomTkinter框架重构UI，提供更现代化的界面风格
- 主窗口尺寸优化为1200x800，提供更好的视觉体验
- 左侧功能区占比25%，右侧预览区占比72%，布局更加合理
- 采用系统主题，支持深色/浅色模式自动切换

### 2.2 人脸库功能

- 在左侧功能区顶部新增人脸库区域
- 支持自动扫描`faces`目录下的jpg/png格式图片
- 采用网格布局，每行最多显示4张人脸缩略图
- 缩略图尺寸统一为48x48像素，保持界面整洁
- 点击任意人脸图片可快速切换为源图片
- 支持动态加载，新增人脸图片无需重启程序

### 2.3 实时换脸功能

- 新增实时换脸按钮，支持摄像头实时预览
- 使用OpenCV进行摄像头画面捕获
- 实时帧率控制在30fps，保证流畅体验
- 支持实时切换源人脸，无需重新启动摄像头
- 提供清晰的状态提示，包括错误信息反馈

### 2.4 功能开关优化

- 统一使用CTkSwitch控件创建开关
- 新增多个实用功能开关：
  - 保持目标FPS
  - 保留临时帧
  - 跳过目标音频
  - 处理多张人脸
- 开关状态实时保存，提升用户体验

## 3. 技术实现

### 3.1 人脸库实现

```python
# 人脸库区域实现
face_gallery_frame = ctk.CTkFrame(left_frame)
face_gallery_frame.place(relx=0.05, rely=0.06, relwidth=0.9, relheight=0.16)
face_images = glob.glob(os.path.join("faces", "*.png")) + glob.glob(
    os.path.join("faces", "*.jpg")
)
max_per_row = 4
thumb_size = (48, 48)
```

### 3.2 实时换脸实现

```python
def start_real_time_face_swap():
    global realtime_running, realtime_cap
    if not roop.globals.source_path:
        update_status("请先选择源人脸图片")
        return

    source_face = get_one_face(cv2.imread(roop.globals.source_path))
    if not source_face:
        update_status("源图片未检测到人脸")
        return

    realtime_cap = cv2.VideoCapture(0)
    if not realtime_cap.isOpened():
        update_status("无法打开摄像头")
        return

    realtime_running = True
    update_status("实时换脸中，点击按钮可停止")
    update_realtime_frame(source_face)
```

## 4. 使用说明

1. 人脸库使用：
   - 将需要的人脸图片放入`faces`目录
   - 支持jpg和png格式
   - 点击任意人脸图片即可切换为源图片

2. 实时换脸：
   - 先选择源人脸图片
   - 点击"实时换脸"按钮启动摄像头
   - 再次点击按钮可停止实时换脸

3. 功能开关：
   - 保持目标FPS：保持视频原有帧率
   - 保留临时帧：保存处理过程中的临时帧
   - 跳过目标音频：不处理音频，加快处理速度
   - 处理多张人脸：同时处理画面中的多个人脸

## 5. 注意事项

1. 实时换脸功能需要摄像头支持
2. 人脸库图片建议使用清晰的正脸照片
3. 实时换脸时建议保持光线充足
4. 处理大尺寸视频时建议开启"跳过目标音频"选项

## 6. 后续优化方向

1. 添加人脸库管理功能（添加、删除、编辑）
2. 优化实时换脸性能
3. 添加更多人脸美化选项
4. 支持批量处理功能
5. 添加处理进度条显示

# Roop

> Take a video and replace the face in it with a face of your choice. You only need one image of the desired face. No dataset, no training.

[![Build Status](https://img.shields.io/github/actions/workflow/status/s0md3v/roop/ci.yml.svg?branch=main)](https://github.com/s0md3v/roop/actions?query=workflow:ci)

<img src="https://i.ibb.co/4RdPYwQ/Untitled.jpg"/>

## Installation

Be aware, the installation needs technical skills and is not for beginners. Please do not open platform and installation related issues on GitHub.

[Basic](https://github.com/s0md3v/roop/wiki/1.-Installation) - It is more likely to work on your computer, but will be quite slow

[Acceleration](https://github.com/s0md3v/roop/wiki/2.-Acceleration) - Unleash the full potential of your CPU and GPU


## Usage

Start the program with arguments:

```
python run.py [options]

-h, --help                                                                 show this help message and exit
-s SOURCE_PATH, --source SOURCE_PATH                                       select an source image
-t TARGET_PATH, --target TARGET_PATH                                       select an target image or video
-o OUTPUT_PATH, --output OUTPUT_PATH                                       select output file or directory
--frame-processor FRAME_PROCESSOR [FRAME_PROCESSOR ...]                    frame processors (choices: face_swapper, face_enhancer, ...)
--keep-fps                                                                 keep target fps
--keep-frames                                                              keep temporary frames
--skip-audio                                                               skip target audio
--many-faces                                                               process every face
--reference-face-position REFERENCE_FACE_POSITION                          position of the reference face
--reference-frame-number REFERENCE_FRAME_NUMBER                            number of the reference frame
--similar-face-distance SIMILAR_FACE_DISTANCE                              face distance used for recognition
--temp-frame-format {jpg,png}                                              image format used for frame extraction
--temp-frame-quality [0-100]                                               image quality used for frame extraction
--output-video-encoder {libx264,libx265,libvpx-vp9,h264_nvenc,hevc_nvenc}  encoder used for the output video
--output-video-quality [0-100]                                             quality used for the output video
--max-memory MAX_MEMORY                                                    maximum amount of RAM in GB
--execution-provider {cpu} [{cpu} ...]                                     available execution provider (choices: cpu, ...)
--execution-threads EXECUTION_THREADS                                      number of execution threads
-v, --version                                                              show program's version number and exit
```


### Headless

Using the `-s/--source`, `-t/--target` and `-o/--output` argument will run the program in headless mode.


## Disclaimer

This software is designed to contribute positively to the AI-generated media industry, assisting artists with tasks like character animation and models for clothing.

We are aware of the potential ethical issues and have implemented measures to prevent the software from being used for inappropriate content, such as nudity.

Users are expected to follow local laws and use the software responsibly. If using real faces, get consent and clearly label deepfakes when sharing. The developers aren't liable for user actions.


## Licenses

Our software uses a lot of third party libraries as well pre-trained models. The users should keep in mind that these third party components have their own license and terms, therefore our license is not being applied.


## Credits

- [deepinsight](https://github.com/deepinsight) for their [insightface](https://github.com/deepinsight/insightface) project which provided a well-made library and models.
- all developers behind the libraries used in this project


## Documentation

Read the [documentation](https://github.com/s0md3v/roop/wiki) for a deep dive.
