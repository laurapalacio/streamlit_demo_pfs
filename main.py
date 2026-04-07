import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
import pandas as pd


class CityData(BaseModel):
    temperature: str
    humidity: str
    image_url: str = Field(description="Double check the URL exists")
    latitude: float
    longitude: float


client = OpenAI()


def main() -> None:
    with st.sidebar:
        st.write("This is the sidebar where you will enter input values :)")
        city_input = st.text_input("Select the city you want to know the weather of:")
        clicked = st.button("Let's go!")

    if clicked:
        st.header(f"This is the city you searched: {city_input}")
        response = client.responses.parse(
            model="gpt-5.4",
            input=[
                {
                    "role": "system",
                    "content": "Extract the weather information and obtain a real image URL.",
                },
                {
                    "role": "user",
                    "content": f"Obtain the most recent weather information for this city {city_input}",
                },
            ],
            text_format=CityData,
        )
        if response.output_parsed is not None:
            st.map(
                pd.DataFrame(
                    [
                        [
                            response.output_parsed.latitude,
                            response.output_parsed.longitude,
                        ]
                    ],
                    columns=["lat", "lon"],
                )
            )
            col1, col2 = st.columns(2)

            with col1:
                st.write("This is the weather forecast from calling the LLM:")
                st.write(f"Temperature: {response.output_parsed.temperature}")
                st.write(f"Humidity: {response.output_parsed.humidity}")

            with col2:
                st.image(response.output_parsed.image_url)
                st.write(f"This is an image of {city_input}")


if __name__ == "__main__":
    main()
