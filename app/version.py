"""The version of this build. Keep it equal to AppVersion in build/h3studio.iss and to the tag of the GitHub release.

Our own builds number above the version they are taken from: the updater offers a release only when its
number is higher, and the releases we install come from the fork named below.
"""

APP_VERSION = "1.2.0"
UPDATE_REPO = "Sewer01/minimax-h3-workspace"   # whose releases this program follows

