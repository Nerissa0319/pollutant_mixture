import os
data_dir = f'{os.path.dirname(os.getcwd())}/data'
saved_dir = f'{os.path.dirname(os.getcwd())}/saved_data'
plot_dir = f'{os.path.dirname(os.getcwd())}/plot'
gpplot_dir = f'{plot_dir}/GP'
# pdplot_mean_dir = f'{gpplot_dir}/partial_dependence_mean'
# pdplot_median_dir = f'{gpplot_dir}/partial_dependence_median'
tdplot_dir = f'{gpplot_dir}/total_dependence'
# irrplot_dir = f'{gpplot_dir}/IRR'
# irr_full = f'{gpplot_dir}/IRR_Full'
# irr_air = f'{gpplot_dir}/IRR_air'
# pdplot_median_dir_air = f'{gpplot_dir}/partial_dependence_median_air'
# pdplot_single_dir = f'{gpplot_dir}/partial_dependence_with_single_varied'
for p in [data_dir, plot_dir, gpplot_dir,
          saved_dir]:
    if not os.path.exists(p):
        os.makedirs(p)