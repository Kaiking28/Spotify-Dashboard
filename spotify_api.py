"""
spotify_api.py

The primary API for interacting with the Spotify songs dataset
for DJ song selection and mixing

"""

import pandas as pd
import numpy as np
import requests


class SpotifyDJAPI:

    def __init__(self):
        self.df = None

    def load_data(self, filename):
        """Load and store the Spotify data"""
        self.df = pd.read_csv(filename)
        self._clean_data()

    def _clean_data(self):
        """Clean and prepare the data for analysis"""
        # Extract year from track_album_release_date
        self.df['year'] = pd.to_datetime(
            self.df['track_album_release_date'],
            errors='coerce'
        ).dt.year

        # Drop rows with missing critical columns
        critical_cols = ['tempo', 'key', 'playlist_genre']
        self.df = self.df.dropna(subset=critical_cols)

        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['track_id'])

        # Convert mode to string (major/minor)
        self.df['mode_name'] = self.df['mode'].map({0: 'Minor', 1: 'Major'})

        # Convert key number to musical key name
        key_map = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                   6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}
        self.df['key_name'] = self.df['key'].map(key_map)
        self.df['full_key'] = self.df['key_name'] + ' ' + self.df['mode_name']

        # Create search string column for autocomplete
        self.df['search_string'] = self.df['track_name'] + ' - ' + self.df['track_artist']

    def get_search_options(self):
        """Get list of search strings for autocomplete"""
        return self.df['search_string'].tolist()

    def get_track_id_from_search(self, search_string):
        """
        Get track ID from search string

        Parameters:
        -----------
        search_string : str, search string in format "track_name - artist_name"

        Returns:
        --------
        str : track_id or None if not found
        """
        if not search_string:
            return None

        result = self.df[self.df['search_string'] == search_string]
        if len(result) == 0:
            return None
        return result.iloc[0]['track_id']

    def get_genres(self):
        """Get list of unique genres"""
        return sorted(self.df['playlist_genre'].unique())

    def get_subgenres(self, genre=None):
        """Get list of unique subgenres, optionally filtered by genre"""
        if genre:
            df = self.df[self.df['playlist_genre'] == genre]
        else:
            df = self.df
        return sorted(df['playlist_subgenre'].unique())

    def get_keys(self):
        """Get list of musical keys"""
        return sorted(self.df['full_key'].unique())

    def get_bpm_range(self):
        """Get the min and max BPM in the dataset"""
        return int(self.df['tempo'].min()), int(self.df['tempo'].max())

    def get_year_range(self):
        """Get the min and max year in the dataset"""
        return int(self.df['year'].min()), int(self.df['year'].max())

    def _filter_by_bpm(self, df, reference_song, bpm_tolerance, include_double_half):
        """
        Filter songs by BPM with optional double/half time matching

        Parameters:
        -----------
        df : DataFrame to filter
        reference_song : Series, reference track data
        bpm_tolerance : int, BPM tolerance in both directions
        include_double_half : bool, whether to include double/half time matches

        Returns:
        --------
        DataFrame with BPM-filtered songs
        """
        bpm_min = reference_song['tempo'] - bpm_tolerance
        bpm_max = reference_song['tempo'] + bpm_tolerance

        if include_double_half:
            double_min = 2 * bpm_min
            double_max = 2 * bpm_max
            half_min = bpm_min / 2
            half_max = bpm_max / 2

            df = df[
                ((df['tempo'] >= bpm_min) & (df['tempo'] <= bpm_max)) |
                ((df['tempo'] >= double_min) & (df['tempo'] <= double_max)) |
                ((df['tempo'] >= half_min) & (df['tempo'] <= half_max))
                ]
        else:
            df = df[(df['tempo'] >= bpm_min) & (df['tempo'] <= bpm_max)]

        return df

    def _filter_by_key(self, df, reference_song, match_key, include_relative):
        """
        Filter songs by musical key with optional relative key matching

        Parameters:
        -----------
        df : DataFrame to filter
        reference_song : Series, reference track data
        match_key : bool, whether to filter by key at all
        include_relative : bool, whether to include relative minor/major keys

        Returns:
        --------
        DataFrame with key-filtered songs
        """
        if not match_key:
            return df

        allowed_keys = [reference_song['full_key']]

        if include_relative:
            key_map = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                       6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}

            ref_key_num = int(reference_song['key'])
            ref_mode = int(reference_song['mode'])

            if ref_mode == 1:  # Major key
                relative_key_num = (ref_key_num + 9) % 12
                relative_key_name = key_map[relative_key_num]
                allowed_keys.append(f"{relative_key_name} Minor")
            else:  # Minor key
                relative_key_num = (ref_key_num + 3) % 12
                relative_key_name = key_map[relative_key_num]
                allowed_keys.append(f"{relative_key_name} Major")

        df = df[df['full_key'].isin(allowed_keys)]
        return df

    def _calculate_similarity(self, df, reference_song):
        """
        Calculate similarity score using KNN Euclidean distance

        Parameters:
        -----------
        df : DataFrame with candidate songs
        reference_song : Series, reference track data

        Returns:
        --------
        DataFrame with added 'distance' and 'similarity' columns
        """
        audio_features = ['danceability', 'energy', 'valence', 'acousticness',
                          'instrumentalness', 'speechiness', 'liveness']

        # Normalize year to 0-1 range
        year_min = df['year'].min()
        year_max = df['year'].max()
        year_range = year_max - year_min

        if year_range > 0:
            year_normalized = (df['year'] - year_min) / year_range
            ref_year_normalized = (reference_song['year'] - year_min) / year_range
        else:
            year_normalized = 0
            ref_year_normalized = 0

        # Calculate squared differences for each feature
        squared_diffs = []
        for feature in audio_features:
            squared_diffs.append((df[feature] - reference_song[feature]) ** 2)
        squared_diffs.append((year_normalized - ref_year_normalized) ** 2)

        # Calculate Euclidean distance
        sum_squared_diffs = sum(squared_diffs)
        df['distance'] = sum_squared_diffs ** 0.5

        # Convert to similarity score (0-1, where 1 is most similar)
        max_distance = df['distance'].max()
        if max_distance > 0:
            df['similarity'] = 1 - (df['distance'] / max_distance)
        else:
            df['similarity'] = 1.0

        return df

    def find_compatible_songs(self, reference_track_id,
                              bpm_tolerance=5,
                              include_double_half=False,
                              match_key=True,
                              include_relative=False,
                              match_genre=False,
                              match_subgenre=False,
                              year_tolerance=10):
        """
        Find songs compatible with a reference track for DJ mixing

        Parameters:
        -----------
        reference_track_id : str, track ID of the reference song
        bpm_tolerance : int, BPM tolerance in both directions (default 5)
        include_double_half : bool, include double/half time matches (default False)
        match_key : bool, filter by musical key (default True)
        include_relative : bool, include relative minor/major keys (default False)
        match_genre : bool, filter by genre (default False)
        match_subgenre : bool, filter by subgenre (default False)
        year_tolerance : int, year tolerance in both directions (default 10)

        Returns:
        --------
        DataFrame with compatible songs sorted by similarity score, or None if reference not found
        """
        # Get reference song
        ref = self.df[self.df['track_id'] == reference_track_id]

        if len(ref) == 0:
            return None

        reference_song = ref.iloc[0]

        # Start with all songs except the reference
        df = self.df[self.df['track_id'] != reference_track_id].copy()

        # Apply filters
        df = self._filter_by_bpm(df, reference_song, bpm_tolerance, include_double_half)
        df = self._filter_by_key(df, reference_song, match_key, include_relative)

        # Genre filtering
        if match_genre:
            df = df[df['playlist_genre'] == reference_song['playlist_genre']]

        # Subgenre filtering
        if match_subgenre:
            df = df[df['playlist_subgenre'] == reference_song['playlist_subgenre']]

        # Year filtering
        year_min = reference_song['year'] - year_tolerance
        year_max = reference_song['year'] + year_tolerance
        df = df[(df['year'] >= year_min) & (df['year'] <= year_max)]

        # Calculate similarity scores
        df = self._calculate_similarity(df, reference_song)

        # Sort by similarity (most similar first)
        df = df.sort_values('similarity', ascending=False)

        return df

    def get_track_by_id(self, track_id):
        """
        Get a single track by its ID

        Parameters:
        -----------
        track_id : str, track ID

        Returns:
        --------
        Series with track data, or None if not found
        """
        result = self.df[self.df['track_id'] == track_id]
        if len(result) == 0:
            return None
        return result.iloc[0]

    def search_deezer_preview(self, track_name: str, artist_name: str):
        """
        Search for a track on Deezer and return the preview URL

        Parameters:
        -----------
        track_name : str, name of the track
        artist_name : str, name of the artist

        Returns:
        --------
        str : Preview URL if found, None otherwise
        """
        query = f"{track_name} {artist_name}"
        url = "https://api.deezer.com/search"

        params = {
            'q': query,
            'limit': 1
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                return data['data'][0].get('preview')
        except:
            pass

        return None