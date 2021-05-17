{{ fullname | escape | underline}}
.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}


{%- block methods %}
{%- if methods %}

{{ _('Methods') | underline(line='-') }}
.. autosummary::
   :toctree:
   :caption: Methods
{% for item in methods %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}


{%- block attributes %}
{%- if attributes %}

{{ _('Attributes') | underline(line='-') }}
.. autosummary::
   :toctree:
   :caption: Attributes
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}
