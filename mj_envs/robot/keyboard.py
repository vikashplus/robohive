# ------------------------------------------------------------------------------
# Keyboard Interface For All OS
# ------------------------------------------------------------------------------

import sys

class _Getch:
  """Gets a single character from standard input.  Does not echo to the
     screen. From http://code.activestate.com/recipes/134892/"""
  def __init__(self):
    try:
      self.impl = _GetchWindows()
    except ImportError:
      try:
        self.impl = _GetchMacCarbon()
      except(AttributeError, ImportError):
        self.impl = _GetchUnix()

  def __call__(self): return self.impl()


class _GetchUnix:
  def __init__(self):
    self._fd = sys.stdin.fileno()

    from termios import tcgetattr
    def get_terminal_settings():
      return tcgetattr(self._fd)
    self._get_terminal_settings = get_terminal_settings

    from tty import setraw
    def set_terminal_raw():
      setraw(self._fd)
    self._set_terminal_raw = set_terminal_raw

    from termios import tcsetattr, TCSADRAIN
    def restore_terminal_settings(settings):
      tcsetattr(self._fd, TCSADRAIN, settings)
    self._restore_terminal_settings = restore_terminal_settings

  def __call__(self):
    old_settings = self._get_terminal_settings()
    try:
      self._set_terminal_raw()
      ch = sys.stdin.read(1)
    finally:
      self._restore_terminal_settings(old_settings)
    return ch


class _GetchWindows:
  def __init__(self):
    import msvcrt
    if sys.version_info[0] == 3:
      def get_char():
        ch = msvcrt.getch()
        if type(ch) is bytes:
          return ch.decode('utf8')
        return ch
      self._getch = get_char
    else:
      self._getch = msvcrt.getch

  def __call__(self):
    ret = self._getch()
    return ret


class _GetchMacCarbon:
  """
  A function which returns the current ASCII key that is down;
  if no ASCII key is down, the null string is returned.  The
  page http://www.mactech.com/macintosh-c/chap02-1.html was
  very helpful in figuring out how to do this.
  """
  def __init__(self):
    import Carbon
    Carbon.Evt #see if it has this (in Unix, it doesn't)
    def has_event_avail():
      return Carbon.Evt.EventAvail(0x0008)[0]!=0
    def get_next_event():
      (what,msg,when,where,mod)=Carbon.Evt.GetNextEvent(0x0008)[1]
      return chr(msg & 0x000000FF)

    self._has_event_avail = has_event_avail
    self._get_next_event = get_next_event

  def __call__(self):
    if self._has_event_avail():
      return self._get_next_event()
    return ''


# Singleton
_getch = _Getch()

def getch():
  return _getch()