#Testing script - orginal script has been changed to yolo_object_detection/scripts
def hyperapp_ui():
    import streamlit as st

    # App configurations
    # apps = [
    #     {"name": "Adverserial Robustness", "url": "http://localhost:8501"},
    #     {"name": "Automating Contracts", "url": "http://localhost:8502"},
    #     {"name": "Churn Prediction", "url": "http://localhost:8503"},
    #     {"name": "Simple Contarstive Learning", "url": "http://localhost:8504"},
    #     {"name": "Object Detection", "url": "http://localhost:8505"},
    # ]

    st.set_page_config(page_title="Master Streamlit App", layout="wide")

    st.title("Master Streamlit Application")
    st.write("Control and view all Streamlit applications from this dashboard.")

    # Navigation bar to list apps
    st.sidebar.header("Applications")
    selected_app = st.sidebar.radio("Select an App", [app["name"] for app in apps])

    # Display the selected app in an iframe
    selected_app_url = next(app["url"] for app in apps if app["name"] == selected_app)

    st.write(f"### Viewing: {selected_app}")
    # st.component.v1.iframe(selected_app_url, width=1200, height=800)

    # Buttons to open apps directly in a new browser tab
    st.write("---")
    st.write("### Quick Access Links")
    for app in apps:
        st.write(f"[{app['name']}]({app['url']})")

if __name__ == "__main__":
    hyperapp_ui()