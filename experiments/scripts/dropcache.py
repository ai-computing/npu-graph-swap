# Drop the page-cache pages of files the caller owns (no root needed): posix_fadvise(DONTNEED).
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
