import streamlit as st
import plotly.graph_objects as go

### we write a function to read a txt file and it's content will be used to populate page with certain information. 
def read_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

file_path = 'src/policies.txt'

# Create a function to check if the user credentials are correct
def authenticate(username, password):
    # Replace this with your own authentication logic
    return username == "user" and password == "password"

def dashboard_page():
    st.sidebar.title('The Conservatives')
    st.sidebar.subheader('Disclaimer: The Conservatives application is a proof-of-concept. It is currently trained on the data for North Dakota for wheat production. In the future, it will be trained for all the states.')
    st.sidebar.markdown('---')
    st.sidebar.text("Please double click")

    # Add spacing
    st.sidebar.markdown("---")

    st.title('The Conservatives')
    st.subheader("Welcome to the application. Please select a state, e.g., North Dakota, to see results for policy.")

    # Add spacing
    st.markdown("---")
    
    st.write("Select State")
    selected_state = st.selectbox('Select State', ['No state selected', 'North Dakota'])

    map = create_map(selected_state)
    st.plotly_chart(map, use_container_width=True)

    st.markdown("Results")
    st.markdown("Select a state to see the results")
    if selected_state == 'North Dakota':
        st.write("North Dakota")
        st.markdown(read_file(file_path=file_path))
    st.markdown("---")
    if st.sidebar.button('Update Model'):
        st.session_state.page = 'authentication'

## define creat map function
def create_map(selected_state):
    if selected_state == 'North Dakota':
        z_values = [1.0]
    else:
        z_values = [0.0]

    fig = go.Figure(data=go.Choropleth(
        locations=['ND'], # Spatial coordinates
        z=z_values, # Data to be color-coded
        locationmode='USA-states', # set of locations match entries in `locations`
        colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(0, 0, 0)']],
        showscale=False,
        hoverinfo='location'
    ))

    fig.update_layout(
        title_text='Highlighting North Dakota',
        geo_scope='usa', # limit map scope to USA
    )

    return fig

# Define the authentication page
def authentication_page():
    st.title("Authentication")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if authenticate(username, password):
                st.success("Logged in as {}".format(username))
                # Run the method for update in the background
                update_method()
                st.session_state.page = 'dashboard'
            else:
                st.error("Invalid credentials")

    with col2:
        if st.button("Cancel"):
            st.session_state.page = 'dashboard'

# Method to update the model in the background
def update_method():
    st.write("Updating model...")

# Run the app
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'

    if st.session_state.page == 'dashboard':
        dashboard_page()
    elif st.session_state.page == 'authentication':
        authentication_page()

if __name__ == "__main__":
    main()
