def show_target_selection(session_data_original, title=None, background_value=None, savedir=None, show=True, add_phase=False):
    import warnings
    # print warning
    warnings.warn('This function is moved. Use popy.plotting.plotting_tools.show_target_selection instead.', DeprecationWarning)
    
    from popy.plotting.plotting_tools import show_target_selection as show_target_selection_new
    show_target_selection_new(session_data_original, title=title, background_value=background_value, savedir=savedir, show=show, add_phase=add_phase)
    