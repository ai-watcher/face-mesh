import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import time
from PIL import Image

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles
DEMO_IMAGE = 'Therapy.jpg'
DEMO_VIDEO = 'Friends.mp4'


st.title("Face Mesh App")

st.markdown(
    """
    <style>
    [data-testid = "stSidebar"][aria-expanded = "true"] > div:first-child{
        width: 350px
    }
    [data-testid = "stSidebar"][aria-expanded = "false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("FaceMesh Sidebar")
st.sidebar.subheader("Parameters")

@st.cache
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = width/float(w)
        dim = (int(w*r), height)

    else:
        r = width/float(w)
        dim = (width, int(h*r))

    #Resize Image
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['Run App on Image',
                                'Run App on Video'])



# For Static Image (Upload Image)
if app_mode == 'Run App on Image':
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    st.sidebar.markdown('----')

    st.markdown(
    """
    <style>
    [data-testid = "stSidebar"][aria-expanded = "true"] > div:first-child{
        width: 400px
    }
    [data-testid = "stSidebar"][aria-expanded = "false"] > div:first-child{
        width: 350px
        margin-left: -400px
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.markdown('**Detected Faces**')
    kpi1_text = st.markdown("0")
    st.markdown('---')

    img_file = st.file_uploader("Upload Your Image",
                                type=['jpg', 'jpeg', 'png'])
    if img_file is not None:
        image = np.array(Image.open(img_file))
    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(DEMO_IMAGE))

    st.sidebar.text("Original Image")
    st.sidebar.image(image)
    face_count = 0

    st.sidebar.markdown('----')
    max_faces = st.sidebar.number_input("Maximum Number of Faces", value=1, min_value=1)
    st.sidebar.markdown('----')
    detection_confidence = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('----')

    #Dashboard
    with mp_face_mesh.FaceMesh(
        static_image_mode = True,
        refine_landmarks = True,
        max_num_faces = max_faces,
        min_detection_confidence = detection_confidence) as face_mesh:
            results = face_mesh.process(image)
            out_image = image.copy()

            #Face Landmark Drawing
            for face_landmarks in results.multi_face_landmarks:
                face_count += 1

                mp_drawing.draw_landmarks(
                image = out_image,
                landmark_list = face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec = drawing_spec,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())

                kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            st.subheader('Output Image')
            st.image(out_image,use_column_width= True)



# For Live Video (Webcam or Upload Video)
elif app_mode == 'Run App on Video':

    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button("Use Webcam")
    record = st.sidebar.checkbox("Record Video")

    if record:
        st.checkbox("Recording", value=True)

    st.markdown(
    """
    <style>
    [data-testid = "stSidebar"][aria-expanded = "true"] > div:first-child{
        width: 400px
    }
    [data-testid = "stSidebar"][aria-expanded = "false"] > div:first-child{
        width: 350px
        margin-left: -400px
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    max_faces = st.sidebar.number_input("Maximum Number of Faces", value=1, min_value=1)
    st.sidebar.markdown('----')
    detection_confidence = st.sidebar.slider('Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('----')
    tracking_confidence = st.sidebar.slider('Minimun Tracking Confidence', min_value=0.0, max_value=1.0, value=0.5)
    st.sidebar.markdown('----')
    
    st.markdown("## Output")

    st_frame = st.empty()
    video_file = st.sidebar.file_uploader("Upload Your Video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file:
        if use_webcam:
            vid = cv2.VideoCapture(0)
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))

    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)
    fps = 0
    i = 0
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Detected Faces**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    #Dashboard
    with mp_face_mesh.FaceMesh(
        refine_landmarks = True,
        max_num_faces = max_faces,
        min_detection_confidence = detection_confidence,
        min_tracking_confidence = tracking_confidence) as face_mesh:

            prevTime = 0
            while vid.isOpened():
                i += 1
                ret, frame = vid.read()
                if not ret:
                    continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            face_count = 0
            if results.multi_face_landmarks:
                #Face Landmark Drawing
                for face_landmarks in results.multi_face_landmarks:
                    face_count += 1
                    mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list = face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec,
                    connection_drawing_spec = drawing_spec)
            
            # Calculate FPS
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            if record:
                #st.checkbox("Recording", value=True)
                out.write(frame)
            #Dashboard
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{face_count}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{width}</h1>", unsafe_allow_html=True)

            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
            frame = image_resize(image = frame, width = 640)
            st_frame.image(frame,channels = 'BGR',use_column_width=True)

    st.text('Video Processed')

    output_video = open('output1.mp4','rb')
    out_bytes = output_video.read()
    st.video(out_bytes)

    vid.release()
    out. release()