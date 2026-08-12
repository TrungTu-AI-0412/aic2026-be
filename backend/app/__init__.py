import av.logging

# PyAV registers a log callback that takes the GIL to forward FFmpeg messages
# into Python logging, from whatever thread emitted them. Our decoders run with
# `thread_type = "AUTO"`, so a worker thread can hit that callback while the
# main thread already holds the GIL inside avcodec_free_context() waiting for
# the same worker to exit: a hard deadlock, and the reason a clips ingestion
# could stall forever in `upserting`. Restoring FFmpeg's own callback sends
# those messages to stderr instead, which is the fix PyAV documents.
#
# Importing av.logging here rather than relying on some later import is what
# makes the ordering safe: PyAV installs its callback when this module is first
# initialised, so the restore has to follow it. `av.logging.set_level()` also
# reinstalls it -- do not call that anywhere.
av.logging.restore_default_callback()
