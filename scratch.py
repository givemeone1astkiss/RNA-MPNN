from rdesign.utils.data import gen_seq_csv, cal_recovery_rate, draw_recovery_scatter

if __name__ == "__main__":
    cal_recovery_rate(
        ref_path='data/ref.csv',
        pred_path='data/submit.csv'
    )
    draw_recovery_scatter(
        recovery_path='data/recovery.csv',
        output_path='data/recovery.png'
    )
