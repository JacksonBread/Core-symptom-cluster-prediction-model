import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import os
import miceforest as mf
import matplotlib.pyplot as plt

# ------------------ GUI 主程序 ------------------
file_path = None
output_folder = None
output_excel_path = None
data = None

# 选择文件
def select_file():
    global file_path
    file_path = filedialog.askopenfilename(
        title="选择 Excel/CSV 文件",
        filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv")]
    )
    if file_path:
        file_label.config(text=f"已选择：{file_path}")
        load_columns()

# 载入列名
def load_columns():
    global data
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    df.columns = df.columns.map(str)  # 转字符串
    data = df.copy()

    listbox_num.delete(0, tk.END)
    for col in data.columns:
        listbox_num.insert(tk.END, col)

# 选择图片输出文件夹
def select_output_folder():
    global output_folder
    output_folder = filedialog.askdirectory(title="选择图像输出文件夹")
    if output_folder:
        folder_label.config(text=f"图片输出位置：{output_folder}")

# 选择 Excel 输出路径
def select_output_excel():
    global output_excel_path
    output_excel_path = filedialog.asksaveasfilename(
        title="选择输出 Excel 文件（包含插补后数据 + 缺失比例表）",
        defaultextension=".xlsx",
        filetypes=[("Excel文件", "*.xlsx")]
    )
    if output_excel_path:
        excel_label.config(text=f"Excel 输出位置：{output_excel_path}")

# 生成【表：变量 × 缺失比例（%）】
def make_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    miss_n = df.isna().sum()
    miss_pct = (miss_n / total * 100.0).round(2)
    out = pd.DataFrame({
        "variable": df.columns,
        "missing_n": miss_n.values,
        "missing_pct(%)": miss_pct.values
    }).sort_values("missing_pct(%)", ascending=False)
    return out

# 开始插补
def start_imputation():
    if data is None:
        messagebox.showerror("错误", "请先选择数据文件")
        return

    num_cols = [listbox_num.get(i) for i in listbox_num.curselection()]
    df = data.copy()

    # 清洗数据
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # 类型转换（连续变量）
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 缺失比例表（无论是否插补都输出）
    missingness_df = make_missingness_table(df)

    missing_columns = [col for col in df.columns if df[col].isnull().any()]
    if not missing_columns:
        # 没缺失值：仍然输出 Missingness 表和原始数据
        if output_excel_path:
            with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
                missingness_df.to_excel(writer, sheet_name="Missingness", index=False)
                df.to_excel(writer, sheet_name="Data_Original", index=False)
        messagebox.showinfo("提示", "没有缺失值，无需插补；已输出 Missingness 表与原始数据。")
        return

    try:
        kernel = mf.ImputationKernel(df, random_state=42)
        kernel.mice(3, verbose=True)
        completed_df = kernel.complete_data(0)

        # 保存 Excel：Missingness + 原始数据 + 插补后数据
        if output_excel_path:
            with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
                missingness_df.to_excel(writer, sheet_name="Missingness", index=False)
                df.to_excel(writer, sheet_name="Data_Original", index=False)
                completed_df.to_excel(writer, sheet_name="Data_Imputed", index=False)

        # 输出每列插补对比图
        if output_folder:
            for col in missing_columns:
                if col in num_cols:
                    plt.figure(figsize=(8, 5))
                    plt.hist(df[col].dropna(), bins=30, density=True, alpha=0.5, label='Before')
                    plt.hist(completed_df[col], bins=30, density=True, alpha=0.5, label='After')
                    plt.xlabel(col)
                    plt.ylabel('Density')
                    plt.title(col)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f"{col}.png"), dpi=300)
                    plt.close()
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    pre_counts = df[col].value_counts(dropna=True)
                    post_counts = completed_df[col].value_counts(dropna=True)

                    # 防止空列报错
                    if len(pre_counts) == 0:
                        axes[0].text(0.5, 0.5, "No observed values", ha="center", va="center")
                        axes[0].set_title(f'{col} - Before')
                    else:
                        axes[0].pie(pre_counts, labels=pre_counts.index, autopct='%1.1f%%')
                        axes[0].set_title(f'{col} - Before')

                    if len(post_counts) == 0:
                        axes[1].text(0.5, 0.5, "No values", ha="center", va="center")
                        axes[1].set_title(f'{col} - After')
                    else:
                        axes[1].pie(post_counts, labels=post_counts.index, autopct='%1.1f%%')
                        axes[1].set_title(f'{col} - After')

                    plt.suptitle(col)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f"{col}.png"), dpi=300)
                    plt.close(fig)

        messagebox.showinfo("完成", "插补已完成！已输出：Missingness 表、插补后 Excel（以及图像如有设置）。")

    except Exception as e:
        messagebox.showerror("错误", f"插补失败：{e}")

# ------------------ GUI 界面 ------------------
root = tk.Tk()
root.title("缺失值插补工具（MICE + miceforest 6.0.4）")
root.geometry("600x620")

btn_file = tk.Button(root, text="选择 Excel/CSV 文件", command=select_file)
btn_file.pack(pady=6)
file_label = tk.Label(root, text="未选择文件")
file_label.pack()

tk.Label(root, text="请选择连续变量（其他默认为分类变量，可多选）：").pack(pady=(12, 4))
listbox_num = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50, height=12)
listbox_num.pack()

btn_folder = tk.Button(root, text="选择图像输出文件夹（可选）", command=select_output_folder)
btn_folder.pack(pady=8)
folder_label = tk.Label(root, text="未选择图片输出位置")
folder_label.pack()

btn_excel = tk.Button(root, text="选择 Excel 输出位置（将包含 Missingness 表）", command=select_output_excel)
btn_excel.pack(pady=8)
excel_label = tk.Label(root, text="未选择 Excel 输出位置")
excel_label.pack()

btn_start = tk.Button(root, text="开始插补", command=start_imputation, bg="green", fg="white")
btn_start.pack(pady=18)

tk.Label(
    root,
    text=(
        "输出 Excel 工作表说明：\n"
        "1) Missingness：变量 × 缺失数量/缺失比例(%)\n"
        "2) Data_Original：清洗后的原始数据\n"
        "3) Data_Imputed：插补后数据（若存在缺失）\n"
    ),
    justify="left"
).pack(pady=10)

root.mainloop()