# from rest_framework import serializers
# from api.models import Data
# #
# #
# # class DataSerializer(serializers.Serializer):
# #     id = serializers.IntegerField(read_only=True)
# #     sick = serializers.BooleanField()
# #     temperature = serializers.FloatField()
# #     heartbeat = serializers.FloatField()
# #     # title = serializers.CharField(required=False, allow_blank=True, max_length=100)
# #     # code = serializers.CharField(style={'base_template': 'textarea.html'})
# #     # linenos = serializers.BooleanField(required=False)
# #     # language = serializers.ChoiceField(choices=LANGUAGE_CHOICES, default='python')
# #     # style = serializers.ChoiceField(choices=STYLE_CHOICES, default='friendly')
# #
# #     def create(self, validated_data):
# #         """
# #         Create and return a new `Data` instance, given the validated data.
# #         """
# #         return Data.objects.create(**validated_data)
# #
# #     def update(self, instance, validated_data):
# #         """
# #         Update and return an existing `Snippet` instance, given the validated data.
# #         """
# #         instance.sick = validated_data.get('sick', instance.sick)
# #         instance.temperature = validated_data.get('temperature', instance.temperature)
# #         instance.heartbeat = validated_data.get('heartbeat', instance.heartbeat)
# #
# #         instance.save()
# #         return instance
#
#
# class DataSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = Data
#         fields = ('id', 'heartbeat', 'temperature', 'sick')
