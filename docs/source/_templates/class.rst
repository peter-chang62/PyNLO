{{ fullname | escape | underline}}

.. contents::
   :local:

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}


{% block attributes -%}
{% if attributes %}
{{ "Attributes" | underline(line="-") }}

.. autosummary::
   :toctree:
{% for item in attributes %}
{%- if item not in inherited_members %}
   ~{{ name }}.{{ item }}
{%- endif -%}
{%- endfor %}
{% endif %}
{%- endblock %}

{% block methods -%}
{% if methods %}
{{ "Methods" | underline(line="-") }}

.. autosummary::
   :toctree:
{% for item in methods %}
{%- if item not in inherited_members %}
   ~{{ name }}.{{ item }}
{%- endif -%}
{%- endfor %}
{% endif %}
{%- endblock %}
