{%- macro hiddentoctree(docnames) -%}
.. toctree::
   :hidden:
{% for docname in docnames %}
   {{ docname }}
{%- endfor %}
{%- endmacro -%}

{{ fullname | escape | underline}}

{%- if ["__init__"]+methods+attributes %}
{{ hiddentoctree(["__init__"]+methods+attributes) }}
{% endif %}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   {{ "Methods" | underline(line="-") }}

   .. autosummary::
      :toctree:
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}

   {% block attributes -%}
   {% if attributes %}
   {{ "Attributes" | underline(line="-") }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {%- endblock %}
