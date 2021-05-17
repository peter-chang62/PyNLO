{{ underline }}
{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}


.. autoclass:: {{ objname }}


{% block methods %}
{% if methods %}
{{ _('Methods') | underline(line='=') }}

.. toctree::
   :hidden:
   :caption: Methods
{% for item in methods %}
   {{ name }}/{{ fullname }}.{{ item }}
{%- endfor %}

.. autosummary::
   :toctree: {{ name }}
{% for item in methods %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block attributes %}
{% if attributes %}
{{ _('Attributes') | underline(line='=') }}

.. toctree::
   :hidden:
   :caption: Attributes
{% for item in attributes %}
   {{ name }}/{{ fullname }}.{{ item }}
{%- endfor %}

.. autosummary::
   :toctree: {{ name }}
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
