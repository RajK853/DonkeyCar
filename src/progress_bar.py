class ProgressBar:
    def __init__(self, total_iter, display_text="Progress", max_bar_size=50, display_interval=20,
                 change_line_at_reset=True, done_char="█", undone_char="░"):
        self.total_iter = total_iter
        self.display_text = display_text
        self.max_bar_size = max_bar_size
        self.display_interval = display_interval
        self.change_line = change_line_at_reset
        self.done_char = done_char
        self.undone_char = undone_char
        self._current_iter = 0

    def set_display_text(self, text):
        assert isinstance(text, str)
        self.display_text = text

    def reset(self):
        self._current_iter = 0
        if self.change_line:
            print()

    def step(self, num=1):
        self._current_iter = min(self._current_iter + num, self.total_iter)
        progress_ratio = self._current_iter/self.total_iter
        bar_size = int(progress_ratio * self.max_bar_size)
        current_bar = bar_size*self.done_char
        if (self._current_iter == self.total_iter) or (not self._current_iter%self.display_interval):
            print(f"\r{self.display_text}: {current_bar:{self.undone_char}<{self.max_bar_size}}"
                  f" {100 * progress_ratio:>6.2f}% ({self._current_iter:>3}/{self.total_iter:<3})", flush=True, end="")
        if self._current_iter == self.total_iter:
            self.reset()
