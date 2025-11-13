import panel as pn
import plotly.graph_objects as go
from spotify_api import SpotifyDJAPI

# Loads javascript dependencies and configures Panel (required)
pn.extension('tabulator', 'plotly')

# Initialize the API!
api = SpotifyDJAPI()
api.load_data("data/spotify_songs.csv")

# GLOBAL STATE
current_song_index = 0
track_id = None
ref = None
filtered_df = None

# WIDGET DECLARATIONS
song_search = pn.widgets.AutocompleteInput(
    name="Search for a song",
    options=api.get_search_options(),
    placeholder="Type to search songs...",
    case_sensitive=False
)
bpm_tolerance = pn.widgets.IntSlider(name="BPM Tolerance (±)", start=0, end=50, value=5, step=1)
year_tolerance = pn.widgets.IntSlider(name="Year Tolerance (±)", start=0, end=100, value=10, step=1)
include_double_half = pn.widgets.Checkbox(name="Include Double/Half BPM", value=False)
include_relative = pn.widgets.Checkbox(name="Include Relative Minor/Major", value=False)
match_key = pn.widgets.Checkbox(name="Match Musical Key", value=False)
match_genre = pn.widgets.Checkbox(name="Match Genre", value=False)
match_subgenre = pn.widgets.Checkbox(name="Match Subgenre", value=False)

prev_button = pn.widgets.Button(name="◀ Previous", button_type="primary", width=140)
next_button = pn.widgets.Button(name="Next ▶", button_type="primary", width=140)

current_song_text = pn.pane.Markdown("", sizing_mode="stretch_width")
audio_player = pn.pane.Audio(name='Preview', sizing_mode="stretch_width")


# CALLBACK FUNCTIONS


def get_key_distribution_wheel(song_search, bpm_tolerance, include_double_half, match_key, include_relative,
                               match_genre, match_subgenre, year_tolerance):
    global track_id, ref, filtered_df

    track_id = api.get_track_id_from_search(song_search)
    if track_id:
        ref = api.get_track_by_id(track_id)
        filtered_df = api.find_compatible_songs(
            track_id, bpm_tolerance, include_double_half, match_key,
            include_relative, match_genre, match_subgenre, year_tolerance
        )
    else:
        ref = None
        filtered_df = None

    reset_index()

    if filtered_df is None or len(filtered_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available")
        return fig

    key_counts = filtered_df['full_key'].value_counts()
    major_keys = ['C Major', 'G Major', 'D Major', 'A Major', 'E Major', 'B Major',
                  'F# Major', 'C# Major', 'G# Major', 'D# Major', 'A# Major', 'F Major']
    minor_keys = ['A Minor', 'E Minor', 'B Minor', 'F# Minor', 'C# Minor', 'G# Minor',
                  'D# Minor', 'A# Minor', 'F Minor', 'C Minor', 'G Minor', 'D Minor']

    all_keys = []
    for i in range(12):
        all_keys.extend([major_keys[i], minor_keys[i]])

    values = [key_counts.get(key, 0) for key in all_keys]
    ref_key = ref['full_key']

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=values,
        theta=[i * 360 / 24 for i in range(24)],
        width=[360 / 24] * 24,
        text=all_keys,
        hovertemplate='<b>%{text}</b><br>Songs: %{r}<extra></extra>',
        showlegend=False,
        marker=dict(color='green')
    ))

    fig.update_layout(
        title=f"Key Distribution - Circle of Fifths (Ref: {ref_key})"
    )
    return fig
def get_reference_info(song_search, bpm_tolerance, include_double_half, match_key, include_relative,
                       match_genre, match_subgenre, year_tolerance):
    global track_id, ref, filtered_df

    if ref is None:
        return "**No song selected**"

    return f"""
    ### Reference Song
    **{ref['track_name']}**  
    *by {ref['track_artist']}*

    - **Album:** {ref['track_album_name']}
    - **BPM:** {ref['tempo']:.1f}
    - **Key:** {ref['full_key']}
    - **Genre:** {ref['playlist_genre']}
    - **Subgenre:** {ref['playlist_subgenre']}
    - **Year:** {int(ref['year'])}
    - **Popularity:** {ref['track_popularity']}/100
    """


def get_year_distribution(song_search, bpm_tolerance, include_double_half, match_key, include_relative,
                          match_genre, match_subgenre, year_tolerance):
    global track_id, ref, filtered_df

    if filtered_df is None or len(filtered_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available")
        return fig

    ref_year = int(ref['year'])
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=filtered_df['year'],
        nbinsx=30,
        name='Compatible Songs',
        hovertemplate='Year: %{x}<br>Songs: %{y}<extra></extra>',
        marker=dict(color='green')
    ))
    fig.add_vline(
        x=ref_year,
        annotation_text=f"Reference ({ref_year})",
        annotation_position="top",
        line=dict(color='black')
    )
    fig.update_layout(
        title=f"Year Distribution (Ref: {ref_year})",
        xaxis_title="Year",
        yaxis_title="Number of Songs"
    )
    return fig


def get_danceability_valence_scatter(song_search, bpm_tolerance, include_double_half, match_key, include_relative,
                                     match_genre, match_subgenre, year_tolerance):
    global track_id, ref, filtered_df

    if filtered_df is None or len(filtered_df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data available")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df['danceability'],
        y=filtered_df['valence'],
        mode='markers',
        name='Matching Songs',
        hovertext=[f"{row['track_name']}\nby {row['track_artist']}" for _, row in filtered_df.iterrows()],
        hoverinfo='text',
        marker=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=[ref['danceability']],
        y=[ref['valence']],
        mode='markers',
        name='Reference Song',
        marker=dict(color='black', size=15)
    ))
    fig.update_layout(
        title="Danceability vs Valence",
        xaxis_title="Danceability",
        yaxis_title="Valence (Happiness)"
    )
    return fig


def get_songs_table(song_search, bpm_tolerance, include_double_half, match_key, include_relative,
                    match_genre, match_subgenre, year_tolerance):
    global track_id, ref, filtered_df

    if filtered_df is None or len(filtered_df) == 0:
        return pn.Column("### No data available")

    display_cols = ['similarity', 'track_name', 'track_artist', 'track_album_name',
                    'tempo', 'full_key', 'playlist_genre', 'track_popularity',
                    'energy', 'danceability', 'year']
    display_df = filtered_df[display_cols].copy()
    display_df['similarity'] = (display_df['similarity'] * 100).round(1)

    table = pn.widgets.Tabulator(
        display_df, page_size=20, pagination='remote', layout='fit_columns', selectable=1,
        sizing_mode='stretch_width',
        formatters={
            'similarity': {'type': 'progress', 'max': 100},
            'tempo': {'type': 'number', 'precision': 1},
            'energy': {'type': 'number', 'precision': 2},
            'danceability': {'type': 'number', 'precision': 2}
        }
    )
    return pn.Column(f"### Found {len(filtered_df)} compatible songs", table, sizing_mode='stretch_width')


# AUDIO PLAYER FUNCTIONS
def update_audio_player():
    global current_song_index, track_id, ref, filtered_df

    track_id = api.get_track_id_from_search(song_search.value)
    if track_id:
        ref = api.get_track_by_id(track_id)
        filtered_df = api.find_compatible_songs(
            track_id, bpm_tolerance.value, include_double_half.value, match_key.value,
            include_relative.value, match_genre.value, match_subgenre.value, year_tolerance.value
        )
    else:
        ref = None
        filtered_df = None

    if filtered_df is None or len(filtered_df) == 0:
        audio_player.object = None
        current_song_text.object = "*No data available*"
        prev_button.disabled = True
        next_button.disabled = True
        return

    if current_song_index >= len(filtered_df):
        current_song_index = len(filtered_df) - 1
    if current_song_index < 0:
        current_song_index = 0

    song = filtered_df.iloc[current_song_index]
    preview_url = api.search_deezer_preview(song['track_name'], song['track_artist'])
    audio_player.object = preview_url

    similarity_pct = song['similarity'] * 100
    current_song_text.object = f"""
        **Now Playing:** {song['track_name']} - {song['track_artist']}  
        *Similarity: {similarity_pct:.1f}% | Song {current_song_index + 1} of {len(filtered_df)}*
        """

    prev_button.disabled = (current_song_index == 0)
    next_button.disabled = (current_song_index >= len(filtered_df) - 1)


def on_prev_click(event):
    global current_song_index
    if current_song_index > 0:
        current_song_index -= 1
    update_audio_player()


def on_next_click(event):
    global current_song_index
    if filtered_df is not None and current_song_index < len(filtered_df) - 1:
        current_song_index += 1
    update_audio_player()


def reset_index():
    global current_song_index
    current_song_index = 0
    update_audio_player()


# ATTACH CALLBACKS
prev_button.on_click(on_prev_click)
next_button.on_click(on_next_click)

# CALLBACK BINDINGS
reference_info = pn.bind(get_reference_info, song_search, bpm_tolerance, include_double_half,
                         match_key, include_relative, match_genre, match_subgenre, year_tolerance)

key_distribution_wheel = pn.bind(get_key_distribution_wheel, song_search, bpm_tolerance,
                                 include_double_half, match_key, include_relative,
                                 match_genre, match_subgenre, year_tolerance)

year_distribution = pn.bind(get_year_distribution, song_search, bpm_tolerance, include_double_half,
                            match_key, include_relative, match_genre, match_subgenre, year_tolerance)

danceability_valence_scatter = pn.bind(get_danceability_valence_scatter, song_search, bpm_tolerance,
                                       include_double_half, match_key, include_relative,
                                       match_genre, match_subgenre, year_tolerance)

songs_table = pn.bind(get_songs_table, song_search, bpm_tolerance, include_double_half,
                      match_key, include_relative, match_genre, match_subgenre, year_tolerance)

# DASHBOARD LAYOUT
search_card = pn.Card(
    pn.Column(
        song_search,
        pn.layout.Divider(),
        reference_info,
        pn.layout.Divider(),
        "**🎵 Preview Compatible Songs:**",
        current_song_text,
        pn.Row(prev_button, next_button),
        audio_player,
        pn.layout.Divider(),
        bpm_tolerance,
        year_tolerance,
        include_double_half,
        match_key,
        include_relative,
        match_genre,
        match_subgenre,
    ),
    title="Find Compatible Songs", width=320, collapsed=False
)

layout = pn.template.FastListTemplate(
    title="Spotify DJ Explorer",
    sidebar=[search_card],
    theme_toggle=False,
    main=[
        pn.Column(
            pn.Row(
                pn.pane.Plotly(key_distribution_wheel, sizing_mode='stretch_width'),
                pn.pane.Plotly(year_distribution, sizing_mode='stretch_width'),
                pn.pane.Plotly(danceability_valence_scatter, sizing_mode='stretch_width')
            ),
            songs_table
        )
    ],
    header_background='#1DB954'
).servable()

layout.show()