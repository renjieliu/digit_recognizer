#!/usr/bin/python3

import cgi
form = cgi.FieldStorage()
img_data = form.getvalue('img_data')

print("Content-Type: text/html")    # HTML is following
print()                             # blank line, end of headers
print("<html><head></head><body>")
print(len(img_data))
print("</body></html>")

