from __future__ import annotations

from audiobooks.llm.schemas import CharacterObservation
from audiobooks.models.character import CharacterRegistryData
from audiobooks.models.script import AudiobookSegment
from audiobooks.text.attribution import apply_speech_tags, name_index, tagged_speaker
from audiobooks.voices.registry import CharacterRegistry, mixed_script, repair_name


def registry() -> CharacterRegistry:
    reg = CharacterRegistry(CharacterRegistryData(book_id="b"))
    for cid, name, aliases, gender in [
        ("rodion", "Раскольников", ["Родион Романович", "молодой человек"], "male"),
        ("alena", "Алена Ивановна", ["старуха", "она"], "female"),
        ("host", "Хозяин", [], "male"),
    ]:
        reg.observe(
            CharacterObservation(id=cid, name=name, aliases=aliases, gender=gender), chapter=1
        )
    return reg


def seg(i: int, kind: str, text: str, speaker: str = "unknown", para: int = 0) -> AudiobookSegment:
    sp = "narrator" if kind == "narration" else speaker
    return AudiobookSegment(index=i, type=kind, speaker=sp, text=text, paragraph=para)


def test_tag_names_speaker_in_nominative_only() -> None:
    index = name_index(registry())
    assert tagged_speaker("громко проговорил хозяин.", index) == "host"
    assert tagged_speaker("отвечал молодой человек, отчасти удивленный", index) == "rodion"
    # the addressee in another case and pronoun aliases never match
    assert tagged_speaker("сказала она Раскольникову", index) is None
    # a remark without a speech verb is not a tag
    assert tagged_speaker("чему особенно способствовала Алена Ивановна", index) is None


def test_apply_speech_tags_fixes_whole_paragraph() -> None:
    segments = [
        seg(0, "dialogue", "Ещё бы!", speaker="rodion", para=1),
        seg(1, "narration", "заметил, зевая, хозяин.", para=1),
        seg(2, "dialogue", "Вот так-то.", speaker="rodion", para=1),
        # narration-first paragraphs are not tag paragraphs
        seg(3, "narration", "Хозяин сказал что-то и ушёл.", para=2),
        seg(4, "dialogue", "Да.", speaker="alena", para=2),
    ]
    changes = apply_speech_tags(segments, registry())
    assert [s.speaker for s in segments] == ["host", "narrator", "host", "narrator", "alena"]
    assert changes == [(0, "rodion", "host"), (2, "rodion", "host")]


def test_mixed_script_name_repair() -> None:
    assert mixed_script("Родion Раскольников")
    assert not mixed_script("Родион Раскольников")
    text = "Родион Романович Раскольников вышел. Родион шёл быстро."
    assert repair_name("Родion Раскольников", text) == "Родион Раскольников"
    # an unrecoverable token is dropped
    assert repair_name("Xyzion Раскольников", "нет подсказок") == "Раскольников"


def test_full_name_merges_with_surname_only_entry() -> None:
    reg = registry()
    cid = reg.observe(
        CharacterObservation(
            id="rodion_raskolnikov", name="Родион Романович Раскольников", gender="male"
        ),
        chapter=2,
    )
    assert cid == "rodion"


def test_reused_id_with_unrelated_name_is_not_merged() -> None:
    reg = registry()
    cid = reg.observe(
        CharacterObservation(id="alena", name="Пульхерия Александровна", gender="female"), 3
    )
    assert cid != "alena"


def test_invented_surname_is_dropped_and_alias_used() -> None:
    from audiobooks.voices.registry import ground_name

    text = "Раскольников, студент, был у вас назад тому месяц, – сказал молодой человек."
    name, aliases = ground_name("Родьон Ромашов", ["Раскольников", "молодой человек"], text)
    assert name == "Раскольников"
    assert aliases == ["молодой человек"]
    # case endings still count as present in the text
    assert ground_name("Раскольников", [], "увидели Раскольникова")[0] == "Раскольников"


def test_description_is_upgraded_to_real_name() -> None:
    reg = CharacterRegistry(CharacterRegistryData(book_id="b"))
    reg.observe(CharacterObservation(id="old_woman", name="Старушка", gender="female"), 1)
    cid = reg.observe(
        CharacterObservation(id="old_woman", name="Алена Ивановна", gender="female"), 1
    )
    assert cid == "old_woman"
    assert reg.get("old_woman").name == "Алена Ивановна"
    assert "Старушка" in reg.get("old_woman").aliases


def test_tag_restores_dialogue_mislabeled_as_narration() -> None:
    from audiobooks.models.script import SegmentType

    segments = [
        seg(0, "narration", "Забавник!", para=3),  # LLM said narration
        seg(1, "narration", "громко проговорил хозяин.", para=3),
    ]
    hints = {0: SegmentType.DIALOGUE, 1: SegmentType.NARRATION}
    apply_speech_tags(segments, registry(), hints)
    assert (segments[0].type, segments[0].speaker) == ("dialogue", "host")
    assert segments[1].speaker == "narrator"


def test_reused_id_with_foreign_name_in_aliases_is_not_merged() -> None:
    reg = registry()
    cid = reg.observe(
        CharacterObservation(
            id="alena", name="Хозяйка", aliases=["Амалия Федоровна"], gender="female"
        ),
        chapter=2,
    )
    assert cid != "alena"
    assert "Амалия Федоровна" not in reg.get("alena").aliases
