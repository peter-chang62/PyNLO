{{ underline }}
{{ fullname | escape | underline}}


.. automodule:: {{ fullname }}


{% block modules %}
{% if modules %}
{{ _('Modules') | underline(line='=')}}

.. toctree::
   :hidden:
   :caption: Modules
{% for item in modules %}
   {{ name }}/{{ item }}
{%- endfor %}
   
.. autosummary::
   :template: module.rst
   :toctree: {{ name }}
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block attributes %}
{% if attributes %}
{{ _('Module Attributes') | underline(line='=')}}

.. toctree::
   :hidden:
   :caption: Module Attributes
{% for item in attributes %}
   {{ name }}/{{ fullname }}.{{ item }}
{%- endfor %}

.. autosummary::
   :toctree: {{ name }}
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block functions %}
{% if functions %}
{{ _('Functions') | underline(line='=') }}

.. toctree::
   :hidden:
   :caption: Functions
{% for item in functions %}
   {{ name }}/{{ fullname }}.{{ item }}
{%- endfor %}

.. autosummary::
   :toctree: {{ name }}
{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block classes %}
{% if classes %}
{{ _('Classes')  | underline(line='=')}}

.. toctree::
   :hidden:
   :caption: Classes
{% for item in classes %}
   {{ name }}/{{ fullname }}.{{ item }}
{%- endfor %}

.. autosummary::
   :toctree: {{ name }}
   :template: class.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block exceptions %}
{% if exceptions %}
{{ _('Exceptions') | underline(line='=') }}

.. toctree::
   :hidden:
   :caption: Exceptions
{% for item in exceptions %}
   {{ name }}/{{ fullname }}.{{ item }}
{%- endfor %}

.. autosummary::
   :toctree: {{ name }}
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
