import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import json


def main():
    try:
        # Step 1: Access the Wikipedia page at the provided URL.
        url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
        response = requests.get(url)
        response.raise_for_status()  # Check for request errors

        # Step 2: Scrape the list of highest grossing films from the page.
        tables = pd.read_html(response.text)
        df = tables[0]  # The first table contains the relevant data

        # Step 3: Filter the data to find movies that grossed over $2 billion and were released before 2020.
        df["Gross"] = df["Gross"].replace({"\$": "", ",": ""}, regex=True).astype(float)
        df["Year"] = df["Release date"].str.extract(r"(\d{4})").astype(int)
        filtered_movies = df[(df["Gross"] > 2_000_000_000) & (df["Year"] < 2020)]

        # Step 4: Count the number of movies that meet the above criteria.
        count_2bn_movies = filtered_movies.shape[0]

        # Step 5: Identify the earliest film that grossed over $1.5 billion from the data.
        earliest_1_5bn_movie = (
            df[df["Gross"] > 1_500_000_000].sort_values("Year").iloc[0]["Title"]
        )

        # Step 6: Calculate the correlation between the Rank and Peak columns.
        correlation = df["Rank"].corr(df["Peak"])

        # Step 7: Create a scatterplot of Rank vs. Peak.
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="Rank", y="Peak")

        # Step 8: Add a dotted red regression line to the scatterplot.
        sns.regplot(
            data=df,
            x="Rank",
            y="Peak",
            scatter=False,
            color="red",
            line_kws={"linestyle": "dotted"},
        )

        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # Step 9: Encode the scatterplot as a base-64 data URI.
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        data_uri = f"data:image/png;base64,{img_str}"

        # Step 10: Compile the answers into a JSON array and include the data URI.
        answers = [
            str(count_2bn_movies),
            earliest_1_5bn_movie,
            str(correlation),
            data_uri,
        ]

        print(json.dumps(answers))

    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
