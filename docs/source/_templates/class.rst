{{ objname | escape | underline}}
.. currentmodule:: {{ module }}
.. autoclass:: {{ objname }}


{%- block methods %}
{%- if methods %}

{{ _('Methods') | underline(line='-') }}
.. autosummary::
   :template: base.rst
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
   :template: base.rst
   :toctree:
   :caption: Attributes
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{%- endif %}
{%- endblock %}
