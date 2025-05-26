from src.strainminer.core import get_data
import pysam as ps
import pytest

@pytest.fixture
def bam_file():
    # Prépare un fichier BAM d'exemple pour les tests
    input_file = ps.AlignmentFile("tests/data/BAMFILE.bam", "rb")
    yield input_file
    input_file.close()

def test_get_data_returns_dict(bam_file):
    contig_name = "ctg0"
    start_pos = 1000
    end_pos = 2000
    result = get_data(bam_file, contig_name, start_pos, end_pos)
    assert isinstance(result, dict), "Le résultat doit être un dictionnaire"

def test_get_data_keys_are_positions(bam_file):
    contig_name = "ctg0"
    start_pos = 1000
    end_pos = 2000
    result = get_data(bam_file, contig_name, start_pos, end_pos)
    for key in result.keys():
        assert isinstance(key, int), f"La clé {key} doit être un entier représentant une position"

def test_get_data_values_are_dicts(bam_file):
    contig_name = "ctg0"
    start_pos = 1000
    end_pos = 2000
    result = get_data(bam_file, contig_name, start_pos, end_pos)
    for value in result.values():
        assert isinstance(value, dict), "La valeur doit être un dictionnaire"

def test_get_data_inner_values_are_0_or_1(bam_file):
    contig_name = "ctg0"
    start_pos = 1000
    end_pos = 2000
    result = get_data(bam_file, contig_name, start_pos, end_pos)
    for value in result.values():
        for sub_value in value.values():
            assert sub_value in (0, 1), "Les valeurs doivent être soit 0 soit 1"

def test_get_data_empty_region_returns_empty_dict(bam_file):
    # Teste une région sans alignements
    contig_name = "ctg0"
    start_pos = 999999
    end_pos = 1000000
    result = get_data(bam_file, contig_name, start_pos, end_pos)
    assert result == {}, "Une région vide doit retourner un dictionnaire vide"

def test_get_data_invalid_contig_raises(bam_file):
    # Teste un contig inexistant
    with pytest.raises(ValueError):
        get_data(bam_file, "invalid_contig", 1000, 2000)

def test_get_data_start_greater_than_end_raises(bam_file):
    # Teste si start > end
    with pytest.raises(ValueError):
        get_data(bam_file, "ctg0", 2000, 1000)