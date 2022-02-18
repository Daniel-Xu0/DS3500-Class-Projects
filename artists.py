"""
Daniel Xu
Homework #1 - Artists & Sankeys
January 26th, 2022
DS3500 - Raichlin
"""
import plotly.graph_objects as go
import pandas as pd
from collections import defaultdict

ARTISTS = 'Artists.json'


# Task #1: Converting JSON to Dataframe
def json_to_df(json_file, columns):
    """ Reads in a json file and turns it into a dataframe with only the specified columns
    :param json_file: name json file (string)
    :return: organized dataframe (df)
    """
    df = pd.read_json(json_file)
    df = df[columns]
    df['Decade'] = df['BeginDate'] - (df['BeginDate'] % 10)
    df.drop('BeginDate', axis=1, inplace=True)
    return df


# Task #2: Aggregate data & Task #3: Clean Rows
def aggregate_df(df, groups):
    """ Count the number of instances of the groups provided
    :param df: a dataframe
    :param groups: what to group the dataframe by (list)
    :return: number of occurrences of all entry types from specified columns (tuple)
    """
    # Create a default dictionary with empty values
    aggregated = defaultdict(int)

    # Group dataframe by Nationality and decade, allowing me to count the number of artists in each group
    grouped = df.groupby(groups).groups
    for key, value in grouped.items():
        aggregated[key] = len(value)
    df = pd.DataFrame.from_dict(aggregated, orient='index', columns=['Artist Count'])

    # Recreate nationality and decade columns, with each combination now corresponding to an artist count
    df.reset_index(inplace=True)
    df[[groups[0], groups[1]]] = pd.DataFrame(df['index'].tolist())
    df.drop(columns='index', inplace=True)

    return df


# Task #3/4: Filter out rows with NA values, decades that are 0, and any rows that have an artist count less than the
# specified value
def clean_df(df, num_artists):
    """ Filter out rows where the decade is 0
    :param df: a dataframe
    :param num_artists: minimum number of artists a nationality and decade must have
    :return: cleaned dataframe
    """
    if 'Decade' in list(df.columns):
        cleaned_df = df[df['Decade'] != 0]
        cleaned_df = cleaned_df[cleaned_df['Artist Count'] >= num_artists]
        cleaned_df.dropna(inplace=True)
        return cleaned_df

    cleaned_df = df[df['Artist Count'] >= num_artists].dropna()
    return cleaned_df


# Task #5: Generate Sankeys
def code_mapping(df, source, target):
    """ Map labels in src and targ columns to integers
    :param df: a dataframe
    :param src: source column
    :param targ: source target
    :return:
    """
    # Sort source and target columns, then add together
    labels = sorted(list(df[source])) + sorted(list(df[target]))
    # Remove duplicates
    labels = list(set(labels))

    # Give labels corresponding integer
    codes = list(range(len(labels)))

    lcmap = dict(zip(labels, codes))

    df = df.replace({source: lcmap, target: lcmap})
    print(df)

    return df, labels


def make_sankey(df, src, targ, vals, node=None):
    """ Make code map for dataframe and draw sankey diagram
    :param df: a dataframe
    :param src: source column
    :param targ: target column
    :param vals: values column
    :param node: node attributes/features (dictionary)
    :return:
    """
    sankey_df, labels = code_mapping(df, src, targ)
    link = {'source': sankey_df[src], 'target': sankey_df[targ], 'value': sankey_df[vals]}
    if not node:
        node = {'pad': 100, 'thickness': 10, 'line':{'color': 'black', 'width': 2}, 'label': labels}
    else:
        node = node
    sk = go.Sankey(link = link, node = node)
    fig = go.Figure(sk)
    fig.show()

def main():
    # Create artists dataframe from JSON
    artists_df = json_to_df(ARTISTS, ["Nationality", "Gender", "BeginDate"])

    # Aggregate dataframe based on nationality and decade
    nationality_vs_decade = aggregate_df(artists_df, ["Nationality", "Decade"])
    nationality_vs_decade = clean_df(nationality_vs_decade, 20)
    make_sankey(nationality_vs_decade, 'Nationality', 'Decade', 'Artist Count')

    # Aggregate dataframe based on gender and decade
    gender_vs_decade = aggregate_df(artists_df, ['Gender', 'Decade'])
    gender_vs_decade = clean_df(gender_vs_decade, 20)
    make_sankey(gender_vs_decade, 'Gender', 'Decade', 'Artist Count')

    # Aggregate dataframe based on nationality and gender
    nationality_vs_gender = aggregate_df(artists_df, ['Nationality', 'Gender'])
    nationality_vs_gender = clean_df(nationality_vs_gender, 20)
    make_sankey(nationality_vs_gender, 'Nationality', 'Gender', 'Artist Count')

main()