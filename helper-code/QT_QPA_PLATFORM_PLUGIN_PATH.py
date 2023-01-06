"""
Use this to import plugin_path if the error message below shows in terminal:
qt.qpa.plugin: Could not find the Qt platform plugin "windows" in ""
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
"""
import os
import PyQt5

dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname, "Qt5", "plugins", "platforms")
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
