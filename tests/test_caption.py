import threading

from trainer.caption import CaptionProcessor


def test_caption_rng_state_roundtrip():
    p = CaptionProcessor(seed=1234, shuffle_caption=True, tag_dropout_rate=0.2)
    p.process("a, b, c, d")
    state = p.get_rng_state()
    # Mutate further, then restore and ensure identical stream afterwards.
    first = [p.process("a, b, c, d") for _ in range(5)]
    p.set_rng_state(state)
    second = [p.process("a, b, c, d") for _ in range(5)]
    assert first == second


def test_caption_process_is_thread_safe():
    """Concurrent process() calls must not corrupt the shared RNG or raise."""
    p = CaptionProcessor(seed=99, shuffle_caption=True, tag_dropout_rate=0.3)
    errors = []

    def worker():
        try:
            for _ in range(200):
                p.process("x, y, z, w")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
