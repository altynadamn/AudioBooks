from __future__ import annotations

from pathlib import Path

from audiobooks.llm.schemas import CharacterObservation
from audiobooks.models.character import CharacterRegistryData
from audiobooks.models.job import GenerationMode
from audiobooks.models.script import AudiobookSegment
from audiobooks.voices.casting import VoiceCaster, resolve_voice
from audiobooks.voices.library import VoiceLibrary
from audiobooks.voices.registry import CharacterRegistry


def obs(**kw: object) -> CharacterObservation:
    return CharacterObservation.model_validate(kw)


def new_registry() -> CharacterRegistry:
    return CharacterRegistry(CharacterRegistryData(book_id="b"))


def test_registry_persistence_roundtrip(tmp_path: Path) -> None:
    reg = new_registry()
    reg.observe(obs(id="anna", name="Анна", aliases=["Аня"], gender="female"), chapter=1)
    path = tmp_path / "character_registry.json"
    reg.save(path)
    loaded = CharacterRegistry.load(path, "b")
    assert loaded.get("anna").aliases == ["Аня"]
    assert loaded.get("anna").gender == "female"
    assert loaded.get("anna").first_chapter == 1


def test_alias_merges_into_existing_character() -> None:
    reg = new_registry()
    reg.observe(obs(id="anna", name="Анна", aliases=["Аня"], gender="female"), 1)
    # a later chunk names her by alias under a different id
    cid = reg.observe(obs(id="anya", name="Аня", gender="female"), 2)
    assert cid == "anna"
    assert len(reg.characters) == 1


def test_same_as_merges_and_adds_alias() -> None:
    reg = new_registry()
    reg.observe(obs(id="anna", name="Анна", gender="female"), 1)
    cid = reg.observe(obs(id="a_s", name="Анна Сергеевна", same_as="anna"), 3)
    assert cid == "anna"
    # the fuller name becomes canonical, the short one stays as an alias
    assert reg.get("anna").name == "Анна Сергеевна"
    assert "Анна" in reg.get("anna").aliases


def test_uncertain_character_is_not_merged() -> None:
    reg = new_registry()
    reg.observe(obs(id="stranger", name="Незнакомец", gender="male"), 1)
    cid = reg.observe(
        obs(id="old_man", name="Старик", gender="male", possibly_same_as=["stranger"]), 2
    )
    assert cid == "old_man"
    assert reg.get("old_man").possibly_same_as == ["stranger"]
    assert len(reg.characters) == 2


def test_gender_conflict_on_reused_id_keeps_characters_apart() -> None:
    reg = new_registry()
    reg.observe(obs(id="sasha", name="Саша", gender="female"), 1)
    cid = reg.observe(obs(id="sasha", name="Саша", gender="male"), 2)
    assert cid != "sasha"
    assert "sasha" in reg.get(cid).possibly_same_as
    # re-applying the same observation (cached chunk on resume) is idempotent
    assert reg.observe(obs(id="sasha", name="Саша", gender="male"), 2) == cid
    assert len(reg.characters) == 2


def test_resolve_speaker_by_name_and_reserved() -> None:
    reg = new_registry()
    reg.observe(obs(id="ivan", name="Иван", aliases=["Ваня"]), 1)
    assert reg.resolve_speaker("Ваня") == "ivan"
    assert reg.resolve_speaker("ivan") == "ivan"
    assert reg.resolve_speaker("narrator") == "narrator"
    assert reg.resolve_speaker("Кто-то другой") == "unknown"


def test_casting_matches_gender_and_spreads_voices(settings) -> None:
    library = VoiceLibrary(settings.voice_library_dir)
    reg = new_registry()
    for cid, name, gender in [
        ("anna", "Анна", "female"),
        ("ivan", "Иван", "male"),
        ("olga", "Ольга", "female"),
        ("petr", "Пётр", "male"),
    ]:
        reg.observe(obs(id=cid, name=name, gender=gender), 1)
    VoiceCaster(library, "narrator_01").cast(reg)
    voices = {c.id: c.voice_id for c in reg.characters.values()}
    assert voices["anna"].startswith("female") and voices["olga"].startswith("female")
    assert voices["ivan"].startswith("male") and voices["petr"].startswith("male")
    assert len(set(voices.values())) == 4  # nobody shares a voice when enough exist
    assert "narrator_01" not in voices.values()


def test_same_character_keeps_voice_across_runs(settings, tmp_path: Path) -> None:
    library = VoiceLibrary(settings.voice_library_dir)
    path = tmp_path / "reg.json"
    reg = new_registry()
    reg.observe(obs(id="anna", name="Анна", gender="female"), 1)
    VoiceCaster(library, "narrator_01").cast(reg)
    first = reg.get("anna").voice_id
    reg.save(path)

    # next chapter: new characters appear, Anna is seen again
    reg2 = CharacterRegistry.load(path, "b")
    reg2.observe(obs(id="anna", name="Анна", gender="female"), 2)
    for i in range(5):
        reg2.observe(obs(id=f"woman_{i}", name=f"Женщина {i}", gender="female"), 2)
    VoiceCaster(library, "narrator_01").cast(reg2)
    assert reg2.get("anna").voice_id == first


def test_pinned_voice_is_respected(settings) -> None:
    library = VoiceLibrary(settings.voice_library_dir)
    reg = new_registry()
    reg.observe(obs(id="anna", name="Анна", gender="female"), 1)
    reg.get("anna").voice_id = "male_02"
    reg.get("anna").voice_locked = True
    VoiceCaster(library, "narrator_01").cast(reg)
    assert reg.get("anna").voice_id == "male_02"


def test_resolve_voice_modes() -> None:
    reg = new_registry()
    reg.observe(obs(id="anna", name="Анна"), 1)
    reg.get("anna").voice_id = "female_01"
    dialogue = AudiobookSegment(index=0, type="dialogue", speaker="anna", text="Привет")
    narration = AudiobookSegment(index=1, type="narration", text="сказала она")
    unknown = AudiobookSegment(index=2, type="dialogue", speaker="unknown", text="Кто там?")
    kw = {"registry": reg, "narrator_voice_id": "narrator_01"}
    assert resolve_voice(dialogue, mode=GenerationMode.CAST, **kw) == "female_01"
    assert resolve_voice(dialogue, mode=GenerationMode.SIMPLE, **kw) == "narrator_01"
    assert resolve_voice(narration, mode=GenerationMode.CAST, **kw) == "narrator_01"
    assert resolve_voice(unknown, mode=GenerationMode.CAST, **kw) == "narrator_01"
    assert (
        resolve_voice(unknown, mode=GenerationMode.CAST, unknown_voice_id="male_01", **kw)
        == "male_01"
    )


def test_merge_redirects_old_ids_and_keeps_pinned_voice() -> None:
    reg = new_registry()
    reg.observe(obs(id="official", name="Чиновник", gender="male"), 2)
    reg.observe(obs(id="marmeladov", name="Мармеладов", gender="male"), 2)
    reg.get("official").voice_id, reg.get("official").voice_locked = "male_02", True
    reg.get("official").line_count, reg.get("marmeladov").line_count = 3, 5
    assert reg.merge("marmeladov", "official")
    assert "official" not in reg.characters
    assert reg.canonical("official") == "marmeladov"
    assert reg.get("official").id == "marmeladov"  # old scripts still resolve
    m = reg.get("marmeladov")
    assert "Чиновник" in m.aliases and m.line_count == 8
    assert (m.voice_id, m.voice_locked) == ("male_02", True)


def test_merge_refuses_contradictions() -> None:
    reg = new_registry()
    reg.observe(obs(id="sasha_f", name="Саша", gender="female"), 1)
    reg.observe(obs(id="petr", name="Пётр", gender="male"), 1)
    assert not reg.merge("sasha_f", "petr")
    assert not reg.merge("petr", "missing")
