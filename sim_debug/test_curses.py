from curses import wrapper
import time
"""
def main(stdscr):
    # Clear screen
    stdscr.clear()

    # This raises ZeroDivisionError when i == 10.
    for i in range(0, 11):
        v = i-10
        time.sleep(0.2)
        stdscr.addstr(i, 0, '10 divided by {} is {}'.format(v, 10/v))
        stdscr.refresh()
        
    stdscr.refresh()
    stdscr.getkey()

wrapper(main)
"""
def print_format_table():
    """
    prints table of formatted text format options
    """
    for style in range(8):
        for fg in range(30,38):
            s1 = ''
            for bg in range(40,48):
                format = ';'.join([str(style), str(fg), str(bg)])
                s1 += '\x1b[%sm %s \x1b[0m' % (format, format)
            print(s1)
        print('\n')

print_format_table()
