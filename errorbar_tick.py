import matplotlib.pyplot as plt


def errorbar_tick(h, w=80, xtype='ratio'):
    """
    Adjusts the width of error bars in a matplotlib errorbar plot.

    Args:
        h (matplotlib.container.ErrorbarContainer): The errorbar plot handle.
        w (float, optional): Error bar width as a ratio of the x-axis
            length (1/w) or in x-axis units if xtype is 'units'.
            Defaults to 80.
        xtype (str, optional):  'ratio' (default) for width as a ratio
            of x-axis length, or 'units' for width in x-axis units. 
    """

    # Get the errorbar data
    if hasattr(h, 'lines'):  # For newer matplotlib versions
        x = h.lines[2][0]._x
    elif hasattr(h, 'children'):  # For older matplotlib versions
        x = h.children[2]._x
    else:
        raise ValueError("Input handle does not seem to be an errorbar plot.")

    # Calculate the error bar width
    if xtype.lower() != 'units':
        ax = plt.gca()
        dx = ax.get_xlim()[1] - ax.get_xlim()[0]
        w = dx / w

    # Adjust the x-data for the errorbar caps
    x[4::9] = x[1::9] - w / 2
    x[7::9] = x[1::9] - w / 2
    x[5::9] = x[1::9] + w / 2
    x[8::9] = x[1::9] + w / 2

    # Update the errorbar plot
    if hasattr(h, 'lines'):  # For newer matplotlib versions
        h.lines[2][0]._x = x
    elif hasattr(h, 'children'):  # For older matplotlib versions
        h.children[2]._x = x