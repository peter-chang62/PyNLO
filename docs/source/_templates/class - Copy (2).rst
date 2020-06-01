{{ fullname | escape | underline}}


.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block methods %}
{% if methods %}
{{ "All Methods" | underline(line="-") }}

.. autosummary::
   :toctree:
{% for item in methods %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block attributes -%}
{% if attributes %}
{{ "All Attributes" | underline(line="-") }}

.. autosummary::
   :toctree:
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{%- endblock %}
