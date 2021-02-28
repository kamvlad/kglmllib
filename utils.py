def duration_str(duration):
    hours = int(duration / 3600)
    duration = duration - hours * 3600
    minutes = int(duration / 60)
    duration = duration - minutes * 60
    seconds = int(duration)
    millis = int((duration - seconds) * 100)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}'