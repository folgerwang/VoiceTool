voices/ -- the actors. CosyVoice is zero-shot: the reference clip IS the voice.

One actor = a PAIR of files with the same name:
    <name>.wav      3-10 s of clean speech, ONE speaker, no music or room noise.
                    Mono. 16 kHz or better (24 kHz / 44.1 kHz fine; the tool
                    resamples). 16-bit or 32-bit float both work.
    <name>.txt      the EXACT transcript of that clip, nothing else. Punctuation
                    as spoken. A wrong transcript gives a wrong voice.

Optional:
    <name>.json     {"work": "作品名"}  - a tag shown in the list, nothing more
    <name>.rhythm   one line: the name of ANOTHER actor whose delivery (pace,
                    stress, phrasing) this one borrows. The line is spoken in
                    that voice first and then converted to this timbre.

Names: letters, digits, _ and - (Chinese is fine). A name starting with "_"
is ignored (scratch).

Picking a clip: choose one that is already close to what you want - the
model copies prosody as well as timbre, so a cheerful clip reads cheerfully
and a flat clip reads flat. No setting makes a voice older, slower or male;
only a different clip does.

storyteller.wav here is CosyVoice's own demo clip (a young female voice),
dropped in so the tool runs end to end. Replace or add your own.
