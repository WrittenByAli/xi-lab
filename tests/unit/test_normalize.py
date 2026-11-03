"""Unit tests for ml/ingest/normalize.py."""

from __future__ import annotations

from ml.ingest.normalize import (
    _infer_format,
    _slugify,
    competition_to_league,
    encode_outcome,
    is_international_match,
)


class TestEncodeOutcome:
    def test_wide_takes_priority(self):
        assert encode_outcome(0, is_wide=True, is_noball=False, is_wicket=False) == "wide"

    def test_noball_after_wide(self):
        assert encode_outcome(0, is_wide=False, is_noball=True, is_wicket=False) == "noball"

    def test_wicket(self):
        assert encode_outcome(0, is_wide=False, is_noball=False, is_wicket=True) == "wicket"

    def test_dot(self):
        assert encode_outcome(0, is_wide=False, is_noball=False, is_wicket=False) == "dot"

    def test_four(self):
        assert encode_outcome(4, is_wide=False, is_noball=False, is_wicket=False) == "4"

    def test_six(self):
        assert encode_outcome(6, is_wide=False, is_noball=False, is_wicket=False) == "6"

    def test_five_maps_to_six(self):
        # 5 is rare (overthrow); we map it to '6'
        assert encode_outcome(5, is_wide=False, is_noball=False, is_wicket=False) == "6"


class TestInferFormat:
    def test_t20(self):
        assert _infer_format("T20") == "t20"

    def test_odi(self):
        assert _infer_format("ODI") == "odi"

    def test_test(self):
        assert _infer_format("Test") == "test"

    def test_unknown_defaults_to_t20(self):
        assert _infer_format("something") == "t20"


class TestSlugify:
    def test_basic(self):
        assert _slugify("Eden Gardens") == "eden_gardens"

    def test_special_chars(self):
        assert _slugify("Gaddafi Stadium, Lahore") == "gaddafi_stadium_lahore"


class TestCompetitionToLeague:
    def test_mapped_mens_league(self):
        r = competition_to_league("Indian Premier League", "T20")
        assert r["league_id"] == "ipl"
        assert r["format"] == "t20"

    def test_unmapped_competition_slugifies_and_infers_format(self):
        # league_id is cosmetic; format must still come from match_type
        # even for a competition name with no league_map.csv entry.
        r = competition_to_league("Some Made-Up Invitational Trophy", "ODI")
        assert r["league_id"] == "some_made_up_invitational_trophy"
        assert r["format"] == "odi"


class TestIsInternationalMatch:
    """is_international is decided from team identity, not competition
    name — competition-name matching (tried first) got this wrong for the
    vast majority of matches: no men's ODI tour name like "Australia tour
    of India" or tournament like "ICC Champions Trophy" ever matched a
    known string, so men's ODI measured ~5% international and men's Test
    measured 0% (every Test is international by definition). See
    ml/ingest/normalize.py's COUNTRY_TEAMS for the full reasoning.
    """

    def test_two_national_teams_is_international(self):
        assert is_international_match("India", "Australia") is True

    def test_bilateral_tour_name_does_not_matter(self):
        # The old, wrong approach kept missing exactly this case.
        assert is_international_match("Australia", "India") is True

    def test_composite_west_indies_team_is_international(self):
        assert is_international_match("West Indies", "England") is True

    def test_domestic_franchise_league_is_not_international(self):
        assert is_international_match("Mumbai Indians", "Chennai Super Kings") is False

    def test_country_vs_franchise_is_not_international(self):
        assert is_international_match("India", "Mumbai Indians") is False

    def test_bare_regional_team_sharing_a_country_name_is_not_international(self):
        # "Barbados" plays only as a West Indian regional/franchise side in
        # this data — the sovereign nation fields no separate international
        # team of its own (that's "West Indies").
        assert is_international_match("Barbados", "Jamaica Tallawahs") is False

    def test_exhibition_team_is_not_international(self):
        assert is_international_match("Asia XI", "Africa XI") is False
