"""Android (Kivy) front-end for efootball-face-converter.

Put face packs in  Download/FaceConverter/input  then press Convert.
Results appear in  Download/FaceConverter/output
"""
import contextlib
import os
import sys
import threading
import traceback
from pathlib import Path

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.togglebutton import ToggleButton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))


def is_android():
    try:
        import android  # noqa: F401
        return True
    except ImportError:
        return False


def work_root():
    if is_android():
        return Path("/storage/emulated/0/Download/FaceConverter")
    return Path.home() / "FaceConverter"


def has_storage_access():
    if not is_android():
        return True
    try:
        from jnius import autoclass
        sdk = autoclass("android.os.Build$VERSION").SDK_INT
        if sdk >= 30:
            return bool(autoclass("android.os.Environment").isExternalStorageManager())
    except Exception:
        pass
    return True


def request_storage_access():
    if not is_android():
        return
    try:
        from jnius import autoclass
        from android import mActivity
        sdk = autoclass("android.os.Build$VERSION").SDK_INT
        if sdk >= 30:
            Intent = autoclass("android.content.Intent")
            Settings = autoclass("android.provider.Settings")
            Uri = autoclass("android.net.Uri")
            intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
            intent.setData(Uri.parse("package:" + mActivity.getPackageName()))
            mActivity.startActivity(intent)
        else:
            from android.permissions import Permission, request_permissions
            request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE])
    except Exception:
        traceback.print_exc()


class LogWriter:
    def __init__(self, callback):
        self.callback = callback

    def write(self, text):
        if text:
            Clock.schedule_once(lambda dt, t=text: self.callback(t))

    def flush(self):
        pass


class FaceApp(App):
    title = "Face Converter"

    def build(self):
        Window.clearcolor = (0.08, 0.08, 0.1, 1)
        root = BoxLayout(orientation="vertical", padding=12, spacing=8)
        self.info = Label(size_hint_y=None, height=110, halign="center", valign="middle")
        self.info.bind(size=lambda w, s: setattr(w, "text_size", (s[0], None)))
        root.add_widget(self.info)

        self.perm_btn = Button(text="1) Grant storage access", size_hint_y=None, height=56)
        self.perm_btn.bind(on_release=lambda *_: self.grant())
        root.add_widget(self.perm_btn)

        self.light = ToggleButton(text="Light mode (smaller texture)", size_hint_y=None, height=48)
        root.add_widget(self.light)

        self.run_btn = Button(text="2) Convert", size_hint_y=None, height=64)
        self.run_btn.bind(on_release=lambda *_: self.start())
        root.add_widget(self.run_btn)

        self.log = Label(text="", size_hint_y=None, halign="left", valign="top")
        self.log.bind(width=lambda w, v: setattr(w, "text_size", (v, None)))
        self.log.bind(texture_size=lambda w, s: setattr(w, "height", max(s[1], 200)))
        scroll = ScrollView()
        scroll.add_widget(self.log)
        root.add_widget(scroll)

        self.running = False
        self.refresh()
        return root

    def on_resume(self):
        self.refresh()
        return True

    def refresh(self):
        r = work_root()
        ok = has_storage_access()
        self.info.text = (
            "Put face files (.zip .cpk .fpk .fmdl .glb or folders) in:\n"
            f"{r / 'input'}\nResults go to:\n{r / 'output'}\n"
            + ("Storage access: OK" if ok else "Storage access: NOT granted")
        )
        self.perm_btn.disabled = ok

    def grant(self):
        request_storage_access()

    def append(self, text):
        self.log.text += text
        if len(self.log.text) > 20000:
            self.log.text = self.log.text[-15000:]

    def start(self):
        if self.running:
            return
        if not has_storage_access():
            self.append("Grant storage access first.\n")
            return
        root = work_root()
        try:
            (root / "input").mkdir(parents=True, exist_ok=True)
            (root / "output").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.append(f"Cannot create folders: {e}\n")
            return
        self.running = True
        self.run_btn.disabled = True
        self.log.text = ""
        threading.Thread(target=self.work, args=(root, self.light.state == "down"), daemon=True).start()

    def work(self, root, light):
        os.environ["FACE_ROOT"] = str(root)
        args = ["batch.py"]
        if light:
            args += ["--max-pixels", "1048576"]
        sys.argv = args
        try:
            import batch
            with contextlib.redirect_stdout(LogWriter(self.append)), contextlib.redirect_stderr(LogWriter(self.append)):
                code = batch.main()
            Clock.schedule_once(lambda dt: self.append(f"\nFinished (code {code}).\n"))
        except BaseException:
            err = traceback.format_exc()
            Clock.schedule_once(lambda dt: self.append("\nERROR:\n" + err))
        finally:
            Clock.schedule_once(lambda dt: self.done())

    def done(self):
        self.running = False
        self.run_btn.disabled = False


if __name__ == "__main__":
    FaceApp().run()
