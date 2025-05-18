import streamlit as st

conversation = {
    'hi': 'Hello! I am HealthBot. How can I assist you today?',
    'hello': 'Hi! Do you have a health-related question?',
    'i have a headache': 'You should rest, stay hydrated, and take a mild pain reliever if needed.',
    'what should i do if i have a fever?': 'Drink plenty of fluids and rest. If the fever persists, please consult a doctor.',
    'i feel dizzy': 'Sit down, breathe deeply, and drink water. If it continues, seek medical help.',
    'what should i eat for a cold?': 'Warm fluids, soups, citrus fruits, and light meals help during a cold.',
    'how to stay healthy?': 'Eat balanced meals, exercise regularly, stay hydrated, and get enough sleep.',
    'what should i do in case of a cut?': 'Clean the wound with water, apply antiseptic, and cover it with a clean bandage.',
    'how much water should i drink daily?': 'Generally, 2 to 3 liters per day is recommended, but it varies based on your activity.',
    'thank you': 'You’re welcome! Take care.',
    'bye': 'Goodbye! Stay healthy.'
}

def get_response(user_input):
    return conversation.get(user_input.lower(), "Sorry, I do not understand that. Can you please ask something else?")

st.title("HealthBot - Healthcare Chatbot")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize user input in session_state if not exists
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

def submit():
    user_input = st.session_state.user_input.strip()
    if user_input:
        response = get_response(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": response})
        st.session_state.user_input = ""  # Clear input box after submit

# Text input with key linked to session_state user_input and submit button
st.text_input("You:", key="user_input", on_change=submit)

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**HealthBot:** {chat['bot']}")
