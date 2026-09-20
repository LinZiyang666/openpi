"""Patch X-WAM evaluation/X-WAM/deploy_policy.py: XWAM_NO_VIDEO=1 skips the 8 extra 3-camera renders per query that only
feed the per-episode mp4 (the policy's own observation render at each query is untouched, so results are identical)."""
import sys
p = sys.argv[1]; s = open(p).read()
old = "        if (ai + 1) % 4 == 0:\n"
new = "        if NO_VIDEO:\n            continue\n        if (ai + 1) % 4 == 0:\n"
assert old in s and "NO_VIDEO" not in s
s = s.replace(old, new, 1)
s = s.replace("import zmq\n", "import zmq\nimport os\nNO_VIDEO = os.environ.get('XWAM_NO_VIDEO', '0') == '1'  # skip the mp4-only renders (ladder speed-up)\n", 1)
old2 = "    gt_videos = np.stack(gt_videos, axis=0)  # [T, V, H, W, 3]\n"
new2 = "    if not gt_videos:\n        return None\n    gt_videos = np.stack(gt_videos, axis=0)  # [T, V, H, W, 3]\n"
assert old2 in s
s = s.replace(old2, new2, 1)
open(p, "w").write(s); print("patched", p)
