import streamlit as st

# Title of the page
st.title("What is the God Particle: The Higgs Boson?")
st.write("")
# Definition of the Higgs Boson
st.write("""
The Higgs Boson is a tiny particle that gives other particles their mass. Often called the 'God Particle,' it helps explain why things in our universe have weight and isn't just floating around as energy.
""")
st.write("")
# Embedding the video
video_url = "https://youtu.be/Ltx2QhRBdHk"
st.video(video_url)

# Section for further information
st.header("Further Reading and Articles")
st.write("""
Here are some articles and resources to dive deeper into the fascinating world of the Higgs Boson:

- [Higgs boson](https://en.wikipedia.org/wiki/Higgs_boson)
- [The Higgs boson: a landmark discovery - ATLAS Experiment](https://atlas.cern/Discover/Physics/Higgs#:~:text=The%20discovery%20of%20the%20Higgs%20boson%20opened%20a%20whole%20new,and%20no%20strong%20force%20interaction.)
- [The Higgs boson, ten years after its discovery](https://home.cern/news/press-release/physics/higgs-boson-ten-years-after-its-discovery)
- [How did we discover the Higgs boson?](https://home.cern/science/physics/higgs-boson/how)
""")

