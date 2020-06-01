{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}

{% block functions %}
{% if functions %}
{{ "Functions" | underline(line="-") }}

.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block classes %}
{% if classes %}
{{ "Classes" | underline(line="-") }}

.. autosummary::
   :toctree:
   :template: class.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block exceptions %}
{% if exceptions %}
{{ "Exceptions" | underline(line="-") }}

.. autosummary::
   :toctree:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
