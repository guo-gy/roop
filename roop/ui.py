import os
import sys
import webbrowser
import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_ALL
from typing import Any, Callable, Tuple, Optional
import cv2
from PIL import Image, ImageOps
import glob

import roop.globals
import roop.metadata
from roop.face_analyser import get_one_face
from roop.capturer import get_video_frame, get_video_frame_total
from roop.face_reference import (
    get_face_reference,
    set_face_reference,
    clear_face_reference,
)
from roop.predictor import predict_frame, clear_predictor
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None

realtime_label = None
realtime_running = False
realtime_cap = None


# todo: remove by native support -> https://github.com/TomSchimansky/CustomTkinter/issues/934
class CTk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_switch(root, text, var, relx, rely, command=None):
    """统一创建开关控件"""
    switch = ctk.CTkSwitch(
        root,
        text=text,
        variable=var,
        cursor="hand2",
        command=command,
    )
    switch.place(relx=relx, rely=rely)
    return switch


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label, realtime_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = CTk()
    root.minsize(1200, 800)
    root.title(f"face face ——guogy")
    root.configure()
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # --- 左侧功能区 ---
    left_frame = ctk.CTkFrame(root)
    left_frame.place(relx=0.01, rely=0.01, relwidth=0.25, relheight=0.98)

    # 人脸库标题
    face_gallery_label = ctk.CTkLabel(
        left_frame, text="人脸库（点击切换源）", anchor="w"
    )
    face_gallery_label.place(relx=0.05, rely=0.01, relwidth=0.9, relheight=0.04)

    # 人脸库区域（自动多行显示，支持jpg/png）
    face_gallery_frame = ctk.CTkFrame(left_frame)
    face_gallery_frame.place(relx=0.05, rely=0.06, relwidth=0.9, relheight=0.16)
    face_images = glob.glob(os.path.join("faces", "*.png")) + glob.glob(
        os.path.join("faces", "*.jpg")
    )
    max_per_row = 4
    thumb_size = (48, 48)
    for idx, face_path in enumerate(face_images):
        img = Image.open(face_path)
        img = ImageOps.contain(img, thumb_size, Image.LANCZOS)  # 动态等比缩放
        tk_img = ctk.CTkImage(img, size=thumb_size)
        btn = ctk.CTkButton(
            face_gallery_frame,
            image=tk_img,
            text="",
            width=thumb_size[0],
            height=thumb_size[1],
            command=lambda p=face_path: select_source_path(p),
        )
        btn.grid(row=idx // max_per_row, column=idx % max_per_row, padx=6, pady=4)

    # 文件选择区

    # 源图片标题
    source_title = ctk.CTkLabel(
        left_frame, text="源图片", anchor="w", font=("微软雅黑", 14, "bold")
    )
    source_title.place(relx=0.05, rely=0.20, relwidth=0.9, relheight=0.05)

    # 源图片显示区
    source_label = ctk.CTkLabel(
        left_frame,
        text="",  # 不要再写文字
        fg_color=ctk.ThemeManager.theme.get("RoopDropArea").get("fg_color"),
        justify="center",
    )
    source_label.place(relx=0.05, rely=0.20, relwidth=0.9, relheight=0.18)

    # 目标图片/视频标题
    target_title = ctk.CTkLabel(
        left_frame, text="目标图片/视频", anchor="w", font=("微软雅黑", 14, "bold")
    )
    target_title.place(relx=0.05, rely=0.42, relwidth=0.9, relheight=0.05)

    # 目标图片/视频显示区
    target_label = ctk.CTkLabel(
        left_frame,
        text="",
        fg_color=ctk.ThemeManager.theme.get("RoopDropArea").get("fg_color"),
        justify="center",
    )
    target_label.place(relx=0.05, rely=0.42, relwidth=0.9, relheight=0.18)

    ctk.CTkButton(
        left_frame, text="选择源", cursor="hand2", command=lambda: select_source_path()
    ).place(relx=0.05, rely=0.38, relwidth=0.9, relheight=0.06)

    ctk.CTkButton(
        left_frame,
        text="选择目标",
        cursor="hand2",
        command=lambda: select_target_path(),
    ).place(relx=0.05, rely=0.60, relwidth=0.9, relheight=0.06)

    # 设置区
    keep_fps_value = ctk.BooleanVar(value=roop.globals.keep_fps)
    create_switch(
        left_frame,
        "保持目标FPS",
        keep_fps_value,
        relx=0.05,
        rely=0.68,
        command=lambda: setattr(roop.globals, "keep_fps", keep_fps_value.get()),
    )
    keep_frames_value = ctk.BooleanVar(value=roop.globals.keep_frames)
    create_switch(
        left_frame,
        "保留临时帧",
        keep_frames_value,
        relx=0.05,
        rely=0.73,
        command=lambda: setattr(roop.globals, "keep_frames", keep_frames_value.get()),
    )
    skip_audio_value = ctk.BooleanVar(value=roop.globals.skip_audio)
    create_switch(
        left_frame,
        "跳过目标音频",
        skip_audio_value,
        relx=0.05,
        rely=0.78,
        command=lambda: setattr(roop.globals, "skip_audio", skip_audio_value.get()),
    )
    many_faces_value = ctk.BooleanVar(value=roop.globals.many_faces)
    create_switch(
        left_frame,
        "处理多张人脸",
        many_faces_value,
        relx=0.05,
        rely=0.83,
        command=lambda: setattr(roop.globals, "many_faces", many_faces_value.get()),
    )

    # 操作按钮区
    ctk.CTkButton(
        left_frame,
        text="开始",
        cursor="hand2",
        command=lambda: select_output_path(start),
    ).place(relx=0.05, rely=0.90, relwidth=0.27, relheight=0.07)
    ctk.CTkButton(
        left_frame, text="预览", cursor="hand2", command=lambda: toggle_preview()
    ).place(relx=0.36, rely=0.90, relwidth=0.27, relheight=0.07)
    ctk.CTkButton(
        left_frame,
        text="实时换脸",
        cursor="hand2",
        command=lambda: toggle_real_time_face_swap(),
    ).place(relx=0.67, rely=0.90, relwidth=0.27, relheight=0.07)

    # --- 右侧预览区 ---
    right_frame = ctk.CTkFrame(root)
    right_frame.place(relx=0.27, rely=0.01, relwidth=0.72, relheight=0.98)

    realtime_label = ctk.CTkLabel(right_frame, text="", width=1, height=1)
    realtime_label.place(relx=0.025, rely=0.025, relwidth=0.95, relheight=0.88)

    status_label = ctk.CTkLabel(right_frame, text="Ready", justify="center")
    status_label.place(relx=0.025, rely=0.92, relwidth=0.95, relheight=0.04)

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    preview.bind("<Up>", lambda event: update_face_reference(1))
    preview.bind("<Down>", lambda event: update_face_reference(-1))
    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def select_source_path(source_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_SOURCE

    if PREVIEW:
        PREVIEW.withdraw()
    if source_path is None:
        source_path = ctk.filedialog.askopenfilename(
            title="select an source image", initialdir=RECENT_DIRECTORY_SOURCE
        )
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        image = render_image_preview(roop.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        roop.globals.source_path = None
        source_label.configure(image=None)


def select_target_path(target_path: Optional[str] = None) -> None:
    global RECENT_DIRECTORY_TARGET

    if PREVIEW:
        PREVIEW.withdraw()
    clear_face_reference()
    if target_path is None:
        target_path = ctk.filedialog.askopenfilename(
            title="select an target image or video", initialdir=RECENT_DIRECTORY_TARGET
        )
    if is_image(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        image = render_image_preview(roop.globals.target_path, (200, 200))
        target_label.configure(image=image)
    elif is_video(target_path):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame)
    else:
        roop.globals.target_path = None
        target_label.configure(image=None)


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if is_image(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save image output file",
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="save video output file",
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        roop.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(roop.globals.output_path)
        start()


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.contain(image, size, Image.LANCZOS)  # 等比缩放完整显示
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.contain(image, size, Image.LANCZOS)  # 等比缩放完整显示
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.unbind("<Right>")
        PREVIEW.unbind("<Left>")
        PREVIEW.withdraw()
        clear_predictor()
    elif roop.globals.source_path and roop.globals.target_path:
        init_preview()
        update_preview(roop.globals.reference_frame_number)
        PREVIEW.deiconify()


def init_preview() -> None:
    PREVIEW.title("Preview [ ↕ Reference face ]")
    if is_image(roop.globals.target_path):
        preview_slider.pack_forget()
    if is_video(roop.globals.target_path):
        video_frame_total = get_video_frame_total(roop.globals.target_path)
        if video_frame_total > 0:
            PREVIEW.title("Preview [ ↕ Reference face ] [ ↔ Frame number ]")
            PREVIEW.bind(
                "<Right>", lambda event: update_frame(int(video_frame_total / 20))
            )
            PREVIEW.bind(
                "<Left>", lambda event: update_frame(int(video_frame_total / -20))
            )
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(roop.globals.reference_frame_number)


def update_preview(frame_number: int = 0) -> None:
    if roop.globals.source_path and roop.globals.target_path:
        temp_frame = get_video_frame(roop.globals.target_path, frame_number)
        if predict_frame(temp_frame):
            sys.exit()
        source_face = get_one_face(cv2.imread(roop.globals.source_path))
        if not get_face_reference():
            reference_frame = get_video_frame(
                roop.globals.target_path, roop.globals.reference_frame_number
            )
            reference_face = get_one_face(
                reference_frame, roop.globals.reference_face_position
            )
            set_face_reference(reference_face)
        else:
            reference_face = get_face_reference()
        for frame_processor in get_frame_processors_modules(
            roop.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                source_face, reference_face, temp_frame
            )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)


def update_face_reference(steps: int) -> None:
    clear_face_reference()
    reference_frame_number = int(preview_slider.get())
    roop.globals.reference_face_position += steps
    roop.globals.reference_frame_number = reference_frame_number
    update_preview(reference_frame_number)


def update_frame(steps: int) -> None:
    frame_number = preview_slider.get() + steps
    preview_slider.set(frame_number)
    update_preview(preview_slider.get())


def toggle_real_time_face_swap():
    global realtime_running, realtime_cap
    if not realtime_running:
        start_real_time_face_swap()
    else:
        stop_real_time_face_swap()


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


def stop_real_time_face_swap():
    global realtime_running, realtime_cap
    realtime_running = False
    if realtime_cap:
        realtime_cap.release()
        realtime_cap = None
    realtime_label.configure(image=None, text="实时换脸预览")  # 停止时显示提示
    update_status("实时换脸已停止")


def update_realtime_frame(source_face):
    global realtime_running, realtime_cap
    if not realtime_running or not realtime_cap:
        return

    ret, frame = realtime_cap.read()
    if not ret:
        update_status("摄像头帧获取失败")
        stop_real_time_face_swap()
        return

    # 检测目标帧中的人脸并进行换脸
    target_face = get_one_face(frame)
    if target_face:
        for frame_processor in get_frame_processors_modules(
            roop.globals.frame_processors
        ):
            frame = frame_processor.process_frame(source_face, target_face, frame)

    # 转为tk可用图片并显示
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.contain(image, (800, 600), Image.LANCZOS)  # 修改预览画面大小
    tk_image = ctk.CTkImage(image, size=image.size)
    realtime_label.configure(image=tk_image)

    # 定时刷新
    realtime_label.after(30, lambda: update_realtime_frame(source_face))
