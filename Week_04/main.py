from PIL import Image
import numpy as np
import cv2
import streamlit as st
import numpy as np
from PIL import Image
from hugchat import hugchat
from hugchat.login import Login

st.title("Title")
st.header("Header")
st.subheader("Subheader")
st.text("# Text")
st.markdown("# Heading 1")
st.markdown("[AI VIETNAM](https://aivietnam.edu.vn/)")
st.markdown("""
1. Machine Learning
2. Deep Learning
""")
st.markdown("$\sqrt{2x+2}$")
st.latex("\sqrt{2x+2}")
st.write('I love AI VIET NAM')
st.write('## Heading 2')
st.write('$ \sqrt{2x+2} $')
st.write('1 + 1 = ', 2)


def get_user_name():
    return 'Quang Tien Nguyen'


with st.echo():
    st.write('This code will be printed.')

    def get_email():
        return 'tienaio2024@gmail.com'
    user_name = get_user_name()
    email = get_email()
    st.write(user_name, email)

st.logo('D:/Onedrive2024/OneDrive/1.0 DS & AI/AIO2024/AIO-Exercise/Module_01/Week_04/source/data/logo.png')

st.image('D:/Onedrive2024/OneDrive/1.0 DS & AI/AIO2024/AIO-Exercise/Module_01/Week_04/source/data/dogs.jpeg',
         caption='Funny dogs.')
st.audio('D:/Onedrive2024/OneDrive/1.0 DS & AI/AIO2024/AIO-Exercise/Module_01/Week_04/source/data/audio.mp4')
st.video('D:/Onedrive2024/OneDrive/1.0 DS & AI/AIO2024/AIO-Exercise/Module_01/Week_04/source/data/video.mp4')


def get_name():
    st.write("Nguyen Quang Tien")


agree = st.checkbox("I agree", on_change=get_name)
if agree:
    st.write("Great!")
st.radio("Your favorite color:", ['Yellow', 'Blue'], captions=['Vàng', 'Xanh'])

option = st.selectbox("Your contact:", ("Email", "Home phone", "Cell phone"))
st.write("Selected:", option)
options = st.multiselect("Your favorite colors:", [
                         "Green", "Yellow", "Red", "Blue"], ["Yellow", "Red"])
st.write("You selected:", options)
color = st.select_slider("Your favorite color:", options=[
                         "red", "orange", "violet"])
st.write("My favorite color is: ", color)

if st.button("Say hello, how are you?"):
    st.write("Hello, how are you today?")
else:
    st.write("Goodbye")
st.link_button("Go to Google", "https://www.google.com.vn/")

title = st.text_input("Movie title:", "Life of Brian")
st.write("The current movie title is", title)

messages = st.container(height=200)
if prompt := st.chat_input("Say something"):
    messages.chat_message("user").write(prompt)
    messages.chat_message("assistant").write(f"Echo: {prompt}")

number = st.number_input("Insert a number")
st.write("The current number is ", number)
values = st.slider("Select a range of values", 0.0, 100.0, (25.0, 75.0))
st.write("Values:", values)

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)

with st.form("my_form"):
    col1, col2 = st.columns(2)
    f_name = col1.text_input('First Name')
    l_name = col2.text_input('Last Name')
    submitted = st.form_submit_button("Submit")
if submitted:
    st.write("First Name: ", f_name, " - Last Name:", l_name)


# Levenshtein distance functions

def levenshtein_distance(token1, token2):
    distances = [[0]*(len(token2)+1) for i in range(len(token1)+1)]

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def load_vocab(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    words = sorted(set([line.strip().lower() for line in lines]))
    return words


vocabs = load_vocab(
    file_path='D:/Onedrive2024/OneDrive/1.0 DS & AI/AIO2024/AIO-Exercise/Module_01/Week_04/source/data/vocab.txt')


# Object detection
MODEL = "D:/Onedrive2024/OneDrive/1.0 DS & AI/AIO2024/AIO-Exercise/Module_01/Week_04/source/model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "D:/Onedrive2024/OneDrive/1.0 DS & AI/AIO2024/AIO-Exercise/Module_01/Week_04/source/model/MobileNetSSD_deploy.prototxt.txt"


def process_image(image):
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (300, 300)), 0.007843, (300, 300), 127.5)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


def annotate_image(image, detections, confidence_threshold=0.5):
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), 70, 2)
    return image


def main():
    # Word Correction using Levenshtein Distance
    st.title("Word Correction using Levenshtein Distance")
    word = st.text_input('Word:')

    if st.button("Compute"):

        # compute levenshtein distance
        leven_distances = dict()
        for vocab in vocabs:
            leven_distances[vocab] = levenshtein_distance(word, vocab)

        # sorted by distance
        sorted_distences = dict(
            sorted(leven_distances.items(), key=lambda item: item[1]))
        correct_word = list(sorted_distences.keys())[0]
        st.write('Correct word: ', correct_word)

        col1, col2 = st.columns(2)
        col1.write('Vocabulary:')
        col1.write(vocabs)

        col2.write('Distances:')
        col2.write(sorted_distences)

    # Object Detection for Images
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")

        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image)
        processed_image = annotate_image(image, detections)
        st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
    print(levenshtein_distance("hel", "hello"))

# App title
st.title('Simple ChatBot')

# Hugging Face Credentials
with st.sidebar:
    st.title('Login HugChat')
    hf_email = st.text_input('Enter E-mail:')
    hf_pass = st.text_input('Enter Password:', type='password')
    if not (hf_email and hf_pass):
        st.warning('Please enter your account!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')


# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function for generating LLM response


def generate_response(prompt_input, email, passwd):
    # Hugging Face Login
    sign = Login(email, passwd)
    cookies = sign.login()
    # Create ChatBot
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    return chatbot.chat(prompt_input)


# User-provided prompt
if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt, hf_email, hf_pass)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
