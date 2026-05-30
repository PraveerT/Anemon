from pathlib import Path


def main():
    p = Path("sidepanel_api/server.py")
    text = p.read_text(encoding="latin-1")
    text = text.replace(
        "                    r'q=(\\d+(?:\\.\\d+)?)\\s+cyc=(\\d+(?:\\.\\d+)?)\\s+tr_acc=(\\d+(?:\\.\\d+)?)%\\s+'\n"
        "                    r'branch=(\\d+(?:\\.\\d+)?)%/(\\d+(?:\\.\\d+)?)%\\s+'\n",
        "                    r'q=(\\d+(?:\\.\\d+)?)\\s+cyc=(\\d+(?:\\.\\d+)?)(?:\\s+rot=(\\d+(?:\\.\\d+)?))?\\s+tr_acc=(\\d+(?:\\.\\d+)?)%\\s+'\n"
        "                    r'branch=(\\d+(?:\\.\\d+)?)%/(\\d+(?:\\.\\d+)?)%\\s+'\n",
    )
    text = text.replace(
        "                    tr_acc = float(m.group(6))\n"
        "                    branch_p1 = float(m.group(7))\n"
        "                    branch_p5 = float(m.group(8))\n"
        "                    fused_p1 = float(m.group(9))\n"
        "                    fused_p5 = float(m.group(10))\n",
        "                    rot_loss = float(m.group(6) or 0.0)\n"
        "                    tr_acc = float(m.group(7))\n"
        "                    branch_p1 = float(m.group(8))\n"
        "                    branch_p5 = float(m.group(9))\n"
        "                    fused_p1 = float(m.group(10))\n"
        "                    fused_p5 = float(m.group(11))\n",
    )
    text = text.replace(
        "                        'cycle_loss': round(cycle_loss, 4),\n"
        "                        'branch_p1': round(branch_p1, 2),\n",
        "                        'cycle_loss': round(cycle_loss, 4),\n"
        "                        'rot_loss': round(rot_loss, 4),\n"
        "                        'branch_p1': round(branch_p1, 2),\n",
    )
    text = text.replace(
        "                        'fusion_w': round(float(m.group(11)), 3),\n"
        "                        'temp_base': round(float(m.group(12)), 3),\n"
        "                        'temp_branch': round(float(m.group(13)), 3),\n",
        "                        'fusion_w': round(float(m.group(12)), 3),\n"
        "                        'temp_base': round(float(m.group(13)), 3),\n"
        "                        'temp_branch': round(float(m.group(14)), 3),\n",
    )
    text = text.replace(
        "                    best_ep = int(m.group(15))\n"
        "                    best_p1 = float(m.group(14))\n",
        "                    best_ep = int(m.group(16))\n"
        "                    best_p1 = float(m.group(15))\n",
    )
    p.write_text(text, encoding="latin-1")
    print("patched rot parser")


if __name__ == "__main__":
    main()
